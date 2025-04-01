################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
# Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
################################################################################

import io
import pickle
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    ReadItem,
)

from bytecheckpoint import BYTECHECKPOINT_GLOBAL_CONFIG
from bytecheckpoint.checkpointer.meta_type import (
    EXTRA_STATE_STR,
    LOADER_CKPT_STR,
    MODEL_STR,
    OPTIMIZER_STR,
    Metadata,
    _StorageInfo,
)
from bytecheckpoint.engine.base_engine import BaseLoadEngine
from bytecheckpoint.planner.default_planner import DefaultLoadPlanner
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger
from bytecheckpoint.utilities.serialization import _create_file_slice, _deserialize_tensor

logger = get_bytecheckpoint_logger()


class LoadEngine(BaseLoadEngine):
    """
    Implementation of LoadEngine using file IO.
    """

    def __init__(self) -> None:
        super().__init__()

    def _slice_file(self, file, sinfo: _StorageInfo):
        return _create_file_slice(file, sinfo.offset, sinfo.length)

    def execute(self, ckpt_name: str, framework_name: str, suffix: Optional[str], *args, **kwargs) -> List[Future[Any]]:
        """
        Executes the appropriate loading function based on the checkpoint name.

        Args:
            ckpt_name (str): The name of the checkpoint, indicating the type of data to load.
            framework_name (str): The name of the framework associated with the checkpoint.
            suffix (Optional[str]): An optional suffix for the checkpoint.
            *args: Additional positional arguments to pass to the loading function.
            **kwargs: Additional keyword arguments to pass to the loading function.

        Returns:
            List[Future[Any]]: A list of futures representing the asynchronous loading tasks.

        Raises:
            ValueError: If the provided ckpt_name is not supported.
        """
        # Select corresponding functions for execution.
        if ckpt_name in {MODEL_STR, OPTIMIZER_STR}:
            return self.load_model_optim_state(ckpt_name, framework_name, suffix, *args, **kwargs)
        elif ckpt_name in {LOADER_CKPT_STR, EXTRA_STATE_STR}:
            return self.load_loader_extra_state(ckpt_name, framework_name, suffix, *args, **kwargs)
        else:
            raise ValueError(f"Do not support {ckpt_name} checkpoint type!")

    """
    Reading Methods.
    """

    def load_metadata(self, metadata_path: str) -> Metadata:
        with open(metadata_path, "rb") as metadata_file:
            metadata = pickle.load(metadata_file)
        return metadata

    def load_model_optim_state(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        relative_path_to_read_items: Dict[str, List[ReadItem]],
        relative_path_file_futures: List[Future[Tuple[str, str]]],
        planner: DefaultLoadPlanner,
        metadata: Metadata,
        fast_loading: bool = False,
    ) -> List[Future[int]]:
        """
        Loads the model and optimizer state from the checkpoint files.

        Args:
            ckpt_name (str): The name of the checkpoint, indicating the type of data to load.
            framework_name (str): The name of the framework associated with the checkpoint.
            suffix (Optional[str]): An optional suffix for the checkpoint.
            relative_path_to_read_items (Dict[str, List[ReadItem]]): A dictionary mapping relative paths to lists of read items.
            relative_path_file_futures (List[Future[Tuple[str, str]]]): A list of futures representing the asynchronous download of checkpoint files.
            planner (DefaultLoadPlanner): The planner used to manage the loading process.
            metadata (Metadata): The metadata associated with the checkpoint.
            fast_loading (bool, optional): Whether to use parallel loading for each file. Defaults to False.

        Returns:
            List[Future[int]]: A list of futures representing the asynchronous loading tasks.
        """
        # Set metadata.
        self.set_metadata(ckpt_name, framework_name, suffix, metadata)

        # Get the relative path to file path mapping.
        relative_path_to_file_path = {}
        # Wait for local checkpoint files.
        logger.info("Waiting for downloading all files for %s_%s", framework_name, ckpt_name)
        download_start_time = time.time()
        for future in as_completed(relative_path_file_futures):
            relative_path, file_path = future.result()
            relative_path_to_file_path[relative_path] = file_path
        download_cost_time = time.time() - download_start_time
        logger.info("End downloading. Total time cost: %s s", download_cost_time)

        # Sync the io futures of the last load event.
        logger.debug("Start waiting for last load events.")
        last_load_start_time = time.time()
        self.sync_io_futures(ckpt_name, framework_name, suffix)
        last_load_time = time.time() - last_load_start_time
        logger.debug("Finish waiting for last load events. Time cost: %s s", last_load_time)

        # Get the worker count and io worker.
        executor = self.get_io_workers(ckpt_name, framework_name, suffix)

        # get metadata.
        storage_data = self.get_storage_data(ckpt_name, framework_name, suffix)
        all_tensor_metadata = self.get_all_tensor_metadata(ckpt_name, framework_name, suffix)

        # Submit per file loading tasks.
        load_futures = []
        for relative_path, reqs in relative_path_to_read_items.items():
            file_path = relative_path_to_file_path[relative_path]
            worker_args = (
                reqs,
                file_path,
                planner,
                storage_data,
                all_tensor_metadata,
            )
            # Per-file parallel loading.
            if fast_loading:
                load_futures.append(executor.submit(load_per_file_pipe, *worker_args))
            # Per-file serial loading.
            else:
                load_per_file_serial(*worker_args)
                future = Future()
                future.set_result(True)
                load_futures.append(future)

        # Record load futures.
        self.set_io_futures(ckpt_name, framework_name, suffix, new_futures=load_futures)

        # Force to sync the loading tasks.
        logger.debug("Start waiting for loading futures")
        future_wait_start = time.time()
        self.sync_io_futures(ckpt_name, framework_name, suffix)
        future_wait_time = time.time() - future_wait_start
        logger.debug("End waiting for loading futures. Time cost: %s s", future_wait_time)
        return load_futures

    def load_loader_extra_state(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        relative_path_file_futures: List[Future[Tuple[str, str]]],
        use_pickle: bool = False,
    ) -> List[Future[Any]]:
        """
        Loads the extra state of the loader from the checkpoint files.

        Args:
            ckpt_name (str): The name of the checkpoint, indicating the type of data to load.
            framework_name (str): The name of the framework associated with the checkpoint.
            suffix (Optional[str]): An optional suffix for the checkpoint.
            relative_path_file_futures (List[Future[Tuple[str, str]]]): A list of futures representing the asynchronous download of checkpoint files.
            use_pickle (bool, optional): Whether to use pickle to load the data. Defaults to False.

        Returns:
            List[Future[Any]]: A list of futures representing the asynchronous loading tasks.
        """
        file_paths = []
        # Wait for local checkpoint files.
        logger.info("Waiting for downloading all files for %s_%s", framework_name, ckpt_name)
        download_start_time = time.time()
        for future in as_completed(relative_path_file_futures):
            _, file_path = future.result()
            file_paths.append(file_path)
        download_cost_time = time.time() - download_start_time
        logger.info("End downloading. Total time cost: %s s", download_cost_time)

        # Sync the io futures of the last load event.
        logger.debug("Start waiting for last load events.")
        last_load_start_time = time.time()
        self.sync_io_futures(ckpt_name, framework_name, suffix)
        last_load_time = time.time() - last_load_start_time
        logger.debug("Finish waiting for last load events. Time cost: %s s", last_load_time)

        # Get the worker count and io worker.
        executor = self.get_io_workers(ckpt_name, framework_name, suffix)

        # Submit per file loading tasks.
        def load_func(file_path, use_pickle):
            if use_pickle:
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            else:
                return torch.load(file_path)

        load_futures = []
        for file_path in file_paths:
            worker_args = (file_path, use_pickle)
            load_futures.append(executor.submit(load_func, *worker_args))

        # Record load futures.
        self.set_io_futures(ckpt_name, framework_name, suffix, new_futures=load_futures)

        # Force to sync the loading tasks.
        logger.debug("Start waiting for loading futures")
        future_wait_start = time.time()
        self.sync_io_futures(ckpt_name, framework_name, suffix)
        future_wait_time = time.time() - future_wait_start
        logger.debug("End waiting for loading futures. Time cost: %s s", future_wait_time)
        return load_futures


def load_per_file_serial(
    reqs: List[ReadItem],
    file_path: str,
    planner: DefaultLoadPlanner,
    storage_data: Dict[MetadataIndex, _StorageInfo],
    all_tensor_metadata: Dict[MetadataIndex, Optional[bytes]],
) -> None:
    """
    Serial load data from a file according to a list of read requests.

    Args:
        reqs (List[ReadItem]): A list of read requests.
        file_path (str): The path to the file to be read.
        planner (DefaultLoadPlanner): The planner for loading data.
        storage_data (Dict[MetadataIndex, _StorageInfo]): A dictionary containing storage information.
        all_tensor_metadata (Dict[MetadataIndex, Optional[bytes]]): A dictionary containing tensor metadata.

    Returns:
        None: This function does not return any value.
    """
    # Sort requests by offset.
    reqs = sorted(reqs, key=lambda req: storage_data[req.storage_index].offset)
    # Perform loading.
    with open(file_path, "rb") as file:
        for req in reqs:
            item_md = storage_data[req.storage_index]
            file_slice = _create_file_slice(file, item_md.offset, item_md.length)
            if req.type == LoadItemType.BYTE_IO:
                bytes = io.BytesIO(file_slice.read(item_md.length))
                bytes.seek(0)
                planner.load_bytes(req, bytes)
            else:
                # Lookup the tensor metadata for deserialization.
                metadata = all_tensor_metadata[req.storage_index]
                assert metadata is not None, f"metadata for tensor: {req.storage_index} is None"
                # Step1: load the storage data.
                storagedata = file_slice.read(item_md.length)
                # Step2: Deserialize the tensor.
                loaded_tensor = _deserialize_tensor(metadata, storagedata)
                loaded_tensor = narrow_tensor_by_index(loaded_tensor, req.storage_offsets, req.lengths)
                target_tensor = planner.resolve_tensor(req).detach()
                assert target_tensor.size() == loaded_tensor.size(), (
                    f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {loaded_tensor.size()}"
                )
                # Step3: H2D copy.
                target_tensor.copy_(loaded_tensor)
                planner.commit_tensor(req, target_tensor)


def load_per_file_pipe(
    reqs: List[ReadItem],
    file_path: str,
    planner: DefaultLoadPlanner,
    storage_data: Dict[MetadataIndex, _StorageInfo],
    all_tensor_metadata: Dict[MetadataIndex, Optional[bytes]],
) -> None:
    """
    Load data from a file using a pipeline with parallel processing.

    This function reads data from a file based on a list of read requests in a pipelined manner.
    It uses a separate thread for reading data from the file and a thread pool for processing the data.

    Args:
        reqs (List[ReadItem]): A list of read requests.
        file_path (str): The path to the file to be read.
        planner (DefaultLoadPlanner): The planner for loading data.
        storage_data (Dict[MetadataIndex, _StorageInfo]): A dictionary containing storage information.
        all_tensor_metadata (Dict[MetadataIndex, Optional[bytes]]): A dictionary containing tensor metadata.

    Returns:
        None: This function does not return any value.
    """
    # Sort requests by offset.
    reqs = sorted(reqs, key=lambda req: storage_data[req.storage_index].offset)
    # Create queues and start the reader thread.
    read_req_queue = queue.Queue()
    return_data_queue = queue.Queue()
    reader_thread = threading.Thread(target=read_worker, args=(read_req_queue, return_data_queue))
    reader_thread.start()
    # Create a thread pool for parallel processing.
    with ThreadPoolExecutor(max_workers=BYTECHECKPOINT_GLOBAL_CONFIG.load_worker_count) as executor:
        futures = []
        with open(file_path, "rb") as file:
            # Submit read tasks to the read_req_queue.
            for req in reqs:
                item_md = storage_data[req.storage_index]
                read_req_queue.put((file, item_md))
            # Process results from return_data_queue.
            for req in reqs:
                return_data = return_data_queue.get()
                # Submit processing task to thread pool.
                tensor_metadata = all_tensor_metadata[req.storage_index]
                worker_args = (req, return_data, planner, tensor_metadata)
                futures.append(executor.submit(load_per_request, *worker_args))
            # Wait for all tasks to complete and collect results
            for future in as_completed(futures):
                future.result()
        # Stop the reader thread
        read_req_queue.put(None)
        reader_thread.join()


def load_per_request(
    req: ReadItem,
    return_data: Union[bytes, torch.Tensor],
    planner: DefaultLoadPlanner,
    tensor_metadata: Optional[bytes] = None,
) -> None:
    """
    Load data for a single read request.

    This function processes a single read request based on its type. If the request type is `LoadItemType.BYTE_IO`,
    it loads the data as bytes. Otherwise, it deserializes the tensor, narrows it according to the request,
    and copies it to the target tensor.

    Args:
        req (ReadItem): The read request to be processed.
        return_data (Union[bytes, torch.Tensor]): The data to be loaded, either as bytes or a tensor.
        planner (DefaultLoadPlanner): The planner responsible for loading data.
        tensor_metadata (Optional[bytes], optional): The metadata for the tensor, required for non-byte IO requests. Defaults to None.

    Raises:
        AssertionError: If the tensor metadata is None for non-byte IO requests, or if the sizes of the target and loaded tensors do not match.

    Returns:
        None: This function does not return any value.
    """
    if req.type == LoadItemType.BYTE_IO:
        bytes_io = io.BytesIO(return_data)
        bytes_io.seek(0)
        planner.load_bytes(req, bytes_io)
    else:
        assert tensor_metadata is not None, f"metadata for tensor: {req.storage_index} is None"
        loaded_tensor = _deserialize_tensor(tensor_metadata, return_data)
        loaded_tensor = narrow_tensor_by_index(loaded_tensor, req.storage_offsets, req.lengths)
        target_tensor = planner.resolve_tensor(req).detach()
        assert target_tensor.size() == loaded_tensor.size(), (
            f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {loaded_tensor.size()}"
        )
        target_tensor.copy_(loaded_tensor)
        planner.commit_tensor(req, target_tensor)


def read_worker(tasks_queue: queue.Queue, result_queue: queue.Queue):
    """
    Continuously reads tasks from a queue, processes them, and puts the results into another queue.

    This function is designed to run in a separate thread. It waits for tasks in the `tasks_queue`,
    each task is a tuple containing a file stream and storage information. It reads the specified
    slice from the file stream and puts the read data into the `result_queue`. The function will
    terminate when it receives a `None` task from the `tasks_queue`.

    Args:
        tasks_queue (queue.Queue): A queue containing tasks to be processed. Each task is a tuple
                                   of a file stream (`io.IOBase`) and storage information (`_StorageInfo`).
        result_queue (queue.Queue): A queue where the results of the tasks will be placed.

    Returns:
        None: This function does not return any value. It runs indefinitely until it receives a `None` task.
    """
    while True:
        task: Optional[Tuple[io.IOBase, _StorageInfo]] = tasks_queue.get()
        if task is None:
            break
        else:
            file_stream, item_md = task
            file_slice = _create_file_slice(file_stream, item_md.offset, item_md.length)
            # Return raw bytes.
            return_data = file_slice.read(item_md.length)
            result_queue.put(return_data)
