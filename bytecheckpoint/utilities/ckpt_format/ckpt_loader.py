################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import collections
import io
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
from torch.distributed.checkpoint.metadata import MetadataIndex

from bytecheckpoint.checkpointer.meta_type import Metadata, _StorageInfo
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger
from bytecheckpoint.utilities.serialization import _deserialize_tensor

logger = get_bytecheckpoint_logger()


class CKPTLoader:
    def __init__(self, url: str, rank: int = None):
        """
        Initialize the CKPTLoader instance.

        Args:
            url (str): The base URL or path where the checkpoint files are located.
            rank (int, optional): The rank of the process. If provided, it is used to create a unique temporary load path. Defaults to None.
        """
        self.url = url
        if rank:
            self.temp_load_path = f"./tmp_merge_{rank}"
        else:
            self.temp_load_path = "./tmp_merge"
        self.meta_data = None

    def load_metadata(self, meta_file: str = ".metadata") -> Metadata:
        """
        Load metadata from a specified file.

        Args:
            meta_file (str): The name of the metadata file. Defaults to ".metadata".

        Returns:
            Metadata: The loaded metadata object.

        Raises:
            Exception: If there is an error loading the metadata from the file.
        """
        meta_url = os.path.join(self.url, meta_file)

        try:
            with open(meta_url, "rb") as metadata_file:
                meta_data = pickle.load(metadata_file)
        except Exception as e:
            logger.error("Failed to load metadata from file %s with exception %s", meta_url, e)
            raise e
        finally:
            # Remove temporary files
            if self.temp_load_path in meta_url:
                os.remove(meta_url)

        self.metadata = meta_data
        return meta_data

    def load_data(
        self,
        relative_path: str,
        offsets: List[int],
        lengths: List[int],
        tensor_metadata: Dict[MetadataIndex, Optional[bytes]],
    ) -> List[Union[torch.Tensor, Any]]:
        """
        Load tensors or objects from a file based on the provided offsets and lengths.

        Args:
            relative_path (str): The relative path to the file containing the tensors or objects.
            offsets (List[int]): A list of offsets indicating where to start reading each tensor or object in the file.
            lengths (List[int]): A list of lengths indicating the size of each tensor or object in the file.
            tensor_metadata (Dict[MetadataIndex, Optional[bytes]]): A dictionary containing metadata for each tensor.

        Returns:
            List[Union[torch.Tensor, Any]]: A list of loaded tensors or objects.

        Raises:
            Exception: If there is an error loading the tensors or objects from the file.
        """
        tensor_url = os.path.join(self.url, relative_path)

        if len(offsets) == 0:
            return
        # combine the continuous chunks into a single one
        chunks = [(offsets[0], lengths[0], 1)]
        for i in range(1, len(offsets)):
            if offsets[i] == offsets[i - 1]:
                assert lengths[i - 1] == 0
            else:
                assert offsets[i] > offsets[i - 1]
            if offsets[i - 1] + lengths[i - 1] == offsets[i]:
                old_offset, old_length, item_num = chunks[-1]
                chunks[-1] = (old_offset, lengths[i] + old_length, item_num + 1)
            else:
                chunks.append((offsets[i], lengths[i], 1))
        res = []
        try:
            openfile = open(tensor_url, "rb")
            idx = 0
            with openfile as f:
                for chunk in chunks:
                    offset, length, item_num = chunk
                    f.seek(offset)
                    data_raw = f.read(length)
                    cur_length = 0
                    for i in range(item_num):
                        cur_data_raw = data_raw[cur_length : cur_length + lengths[idx]]
                        cur_metadata = tensor_metadata[idx]
                        # Byte object.
                        if cur_metadata is None:
                            cur_data_bytes = io.BytesIO(cur_data_raw)
                            cur_data = torch.load(cur_data_bytes)
                        # Tensor object.
                        else:
                            cur_data = _deserialize_tensor(cur_metadata, cur_data_raw)
                        res.append(cur_data)
                        cur_length += lengths[idx]
                        idx += 1
        except Exception as e:
            logger.error("Failed to load tensor from file %s with exception %s", relative_path, e)
            raise e
        finally:
            # Remove temporary files
            if self.temp_load_path in tensor_url:
                os.remove(tensor_url)
        return res


def distcp_load_tool(
    ckpt_path: str,
    relative_path: str,
) -> Dict[str, List[Dict[str, Union[Tuple[int], torch.Tensor, Any]]]]:
    """
    Load tensors or objects from a checkpoint file using the provided checkpoint path and relative path.

    Args:
        ckpt_path (str): The base path where the checkpoint files are located.
        relative_path (str): The relative path to the file containing the tensors or objects.

    Returns:
        Dict[str, List[Dict[str, Union[Tuple[int], torch.Tensor, Any]]]]: A dictionary where the keys are tensor names,
        and the values are lists of dictionaries containing the offset and the corresponding tensor or object.
    """
    loader = CKPTLoader(ckpt_path)

    metadata = loader.load_metadata()

    sd = metadata.storage_data
    tensor_metadata = metadata.all_tensor_metadata
    targets = [key for key in sd if sd[key].relative_path == relative_path]
    targets.sort(key=lambda x: sd[x].offset)
    offsets = [sd[key].offset for key in targets]
    lengths = [sd[key].length for key in targets]
    tensor_keys = [key.fqn for key in targets]
    tensor_nd_offset = [key.offset for key in targets]
    tensor_metadata = [tensor_metadata[key] for key in targets]
    tensors_or_objects = loader.load_data(relative_path, offsets, lengths, tensor_metadata)
    assert len(tensors_or_objects) == len(tensor_metadata)
    # Key: tensor name
    # Value: A list of kv pairs: {"offset": Tuple, "tensor_or_object": torch.Tensor or Objects}
    state_dict: Dict[str, List[Dict[str, Union[Tuple[int], torch.Tensor, Any]]]] = {}
    assert len(tensor_keys) == len(tensors_or_objects)
    for idx, key in enumerate(tensor_keys):
        if key not in state_dict:
            state_dict[key] = []
        state_dict[key].append({"offset": tensor_nd_offset[idx], "tensor_or_object": tensors_or_objects[idx]})
    return state_dict


def load_tensor_shards_single_file(
    storage_data: Dict[MetadataIndex, _StorageInfo],
    all_tensor_metadata: Dict[MetadataIndex, Optional[bytes]],
    loader: CKPTLoader,
    file: str,
    dtype: Optional[torch.dtype],
) -> Tuple[Dict[str, Any], Dict[str, List[Tuple[torch.Tensor, Tuple[int]]]]]:
    """
    Load tensor shards from a single file.

    This function processes a single file to load tensor shards based on the provided storage data and metadata.
    It extracts relevant information from the storage data, such as offsets and lengths, and uses the loader to
    retrieve the tensor or object data. The results are then organized into a final state dictionary and a dictionary
    of tensor shards.

    Args:
        storage_data (Dict[MetadataIndex, _StorageInfo]): A dictionary containing storage information for each tensor.
        all_tensor_metadata (Dict[MetadataIndex, Optional[bytes]]): A dictionary containing metadata for each tensor.
        loader (CKPTLoader): An instance of CKPTLoader used to load the tensor data.
        file (str): The path to the file containing the tensor shards.
        dtype (Optional[torch.dtype]): The data type of the tensors.

    Returns:
        Tuple[Dict[str, Any], Dict[str, List[Tuple[torch.Tensor, Tuple[int]]]]]: A tuple containing the final state dictionary
        and a dictionary of tensor shards, where keys are tensor names and values are lists of tuples containing the
        tensor and its offset.
    """
    tensor_shards_kv = collections.defaultdict(list)
    final_state_dict = {}
    start_time = time.time()
    relative_path = file.split("/")[-1]
    targets = [i for i in storage_data if storage_data[i].relative_path == relative_path]
    if len(targets) == 0:
        logger.info("No tensor in this discp file. Skip")
        return final_state_dict, tensor_shards_kv
    targets.sort(key=lambda x: storage_data[x].offset)
    file_offsets = [storage_data[i].offset for i in targets]
    lengths = [storage_data[i].length for i in targets]
    tensor_metadata = [all_tensor_metadata[i] for i in targets]
    tensor_keys = [i.fqn for i in targets]
    tensor_offsets = [k.offset for k in targets]

    if tensor_keys:
        tensors_or_objects = loader.load_data(relative_path, file_offsets, lengths, tensor_metadata)
        if tensors_or_objects is not None:
            assert len(tensor_keys) == len(tensors_or_objects)
            for idx, tensor_or_object in enumerate(tensors_or_objects):
                # Byte object.
                if not isinstance(tensor_or_object, torch.Tensor):
                    final_state_dict[tensor_keys[idx]] = tensor_or_object
                # Tensor object.
                else:
                    if dtype:
                        tensor_shards_kv[tensor_keys[idx]].append((tensor_or_object.to(dtype), tensor_offsets[idx]))
                    else:
                        tensor_shards_kv[tensor_keys[idx]].append((tensor_or_object, tensor_offsets[idx]))

        logger.info("process file: %s, cost time: %s s", file, time.time() - start_time)
    else:
        logger.info("no tensor to read in file: %s, cost time: %s s", file, time.time() - start_time)
    return final_state_dict, tensor_shards_kv


def load_tensor_shards_non_async(
    storage_data: Dict[MetadataIndex, _StorageInfo],
    all_tensor_metadata: Dict[MetadataIndex, Optional[bytes]],
    loader: CKPTLoader,
    files: List[str],
    dtype: Optional[torch.dtype],
) -> Tuple[Dict[str, Any], Dict[str, List[Tuple[torch.Tensor, Tuple[int]]]]]:
    """
    Load tensor shards from multiple files in a non-async manner.

    This function iterates over a list of files and loads tensor shards from each file using the `load_tensor_shards_single_file` function.
    The results are then aggregated into a final state dictionary and a dictionary of tensor shards.

    Args:
        storage_data (Dict[MetadataIndex, _StorageInfo]): A dictionary containing storage information for each tensor.
        all_tensor_metadata (Dict[MetadataIndex, Optional[bytes]]): A dictionary containing metadata for each tensor.
        loader (CKPTLoader): An instance of CKPTLoader used to load the tensor data.
        files (List[str]): A list of file paths to load tensor shards from.
        dtype (Optional[torch.dtype]): The data type of the tensors.

    Returns:
        Tuple[Dict[str, Any], Dict[str, List[Tuple[torch.Tensor, Tuple[int]]]]]: A tuple containing the final state dictionary
        and a dictionary of tensor shards, where keys are tensor names and values are lists of tuples containing the
        tensor and its offset.
    """
    tensor_shards_kv = collections.defaultdict(list)
    final_state_dict = {}
    for file in files:
        final_state_dict_per_file, tensor_shards_kv_per_file = load_tensor_shards_single_file(
            storage_data,
            all_tensor_metadata,
            loader,
            file,
            dtype,
        )
        final_state_dict.update(final_state_dict_per_file)
        for k, v in tensor_shards_kv_per_file.items():
            tensor_shards_kv[k].extend(v)
    return final_state_dict, tensor_shards_kv


def load_tensor_shards_multiprocessing_execute(
    storage_data: Dict[MetadataIndex, _StorageInfo],
    all_tensor_metadata: Dict[MetadataIndex, Optional[bytes]],
    loader: CKPTLoader,
    queue_in: mp.Queue,
    queue_out: mp.Queue,
    queue_exit: mp.Queue,
    dtype: Optional[torch.dtype],
) -> None:
    """
    Execute the process of loading tensor shards from multiple files using multiprocessing.

    This function is designed to be run in a separate process. It continuously fetches file paths from the input queue,
    loads tensor shards from these files, and puts the results into the output queue. It terminates when it receives a
    sentinel value (None) from the input queue.

    Args:
        storage_data (Dict[MetadataIndex, _StorageInfo]): A dictionary containing storage information for each tensor.
        all_tensor_metadata (Dict[MetadataIndex, Optional[bytes]]): A dictionary containing metadata for each tensor.
        loader (CKPTLoader): An instance of CKPTLoader used to load the tensor data.
        framework (str): The framework used for loading the tensors, e.g., "fsdp".
        queue_in (mp.Queue): The input queue from which file paths are fetched.
        queue_out (mp.Queue): The output queue into which the results (final state dictionary and tensor shards) are put.
        queue_exit (mp.Queue): The exit queue used to signal the process to exit.
        dtype (Optional[torch.dtype]): The data type of the tensors.

    Returns:
        None
    """
    while True:
        file = queue_in.get()
        if file is None:  # check for sentinel value
            break
        if file.endswith(".metadata"):
            continue
        final_state_dict, tensor_shards_kv = load_tensor_shards_single_file(
            storage_data,
            all_tensor_metadata,
            loader,
            file,
            dtype,
        )
        queue_out.put((final_state_dict, tensor_shards_kv))

    queue_out.put((None, None))
    queue_exit.get()


def load_tensor_shards_multiprocessing(
    storage_data: Dict[MetadataIndex, _StorageInfo],
    all_tensor_metadata: Dict[MetadataIndex, Optional[bytes]],
    loader: CKPTLoader,
    files: List[str],
    num_io_workers: int,
    dtype: Optional[torch.dtype],
) -> Tuple[Dict[str, Any], Dict[str, List[Tuple[torch.Tensor, Tuple[int]]]]]:
    """
    Load tensor shards from multiple files using multiprocessing.

    This function uses multiple processes to load tensor shards from a list of files in parallel.
    Each process fetches a file from the input queue, loads tensor shards from it, and puts the results into the output queue.
    The final state dictionary and tensor shards are aggregated from the results of all processes.

    Args:
        storage_data (Dict[MetadataIndex, _StorageInfo]): A dictionary containing storage information for each tensor.
        all_tensor_metadata (Dict[MetadataIndex, Optional[bytes]]): A dictionary containing metadata for each tensor.
        loader (CKPTLoader): An instance of CKPTLoader used to load the tensor data.
        files (List[str]): A list of file paths to load tensor shards from.
        num_io_workers (int): The number of worker processes to use for loading tensor shards.
        dtype (Optional[torch.dtype]): The data type of the tensors.

    Returns:
        Tuple[Dict[str, Any], Dict[str, List[Tuple[torch.Tensor, Tuple[int]]]]]: A tuple containing the final state dictionary
        and a dictionary of tensor shards, where keys are tensor names and values are lists of tuples containing the
        tensor and its offset.
    """
    # Loading tensor shards from multiple files using multiprocessing
    tensor_shards_kv = collections.defaultdict(list)
    final_state_dict = {}
    processes = []
    queue_in = mp.Queue()
    queue_exit = mp.Queue()
    queue_out = [mp.Queue() for _ in range(num_io_workers)]
    for i in range(num_io_workers):
        p = mp.Process(
            target=load_tensor_shards_multiprocessing_execute,
            args=(
                storage_data,
                all_tensor_metadata,
                loader,
                queue_in,
                queue_out[i],
                queue_exit,
                dtype,
            ),
        )
        p.start()
        processes.append(p)
    for file in files:
        queue_in.put(file)
    for _ in range(num_io_workers):
        queue_in.put(None)
    logger.info("start waiting for process queue")
    for i in range(num_io_workers):
        while True:
            get_start_time = time.time()
            final_state_dict_per_file, tensor_shards_kv_per_file = queue_out[i].get()
            if final_state_dict_per_file is None and tensor_shards_kv_per_file is None:
                break
            logger.info(
                "wait for getting tensor shards from process %s, cost time: %s", i, time.time() - get_start_time
            )
            final_state_dict.update(final_state_dict_per_file)
            for k, v in tensor_shards_kv_per_file.items():
                tensor_shards_kv[k].extend(v)
    for _ in range(num_io_workers):
        queue_exit.put(None)
    for p in processes:
        p.join()  # wait for all subprocesses to finish
    return final_state_dict, tensor_shards_kv
