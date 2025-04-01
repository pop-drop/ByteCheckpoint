################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in storing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import io
import os
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.planner import (
    SavePlan,
    WriteItem,
    WriteItemType,
)

from bytecheckpoint import BYTECHECKPOINT_GLOBAL_CONFIG
from bytecheckpoint.checkpointer.meta_type import (
    DEFAULT_SUFFIX,
    EXTRA_STATE_STR,
    LOADER_CKPT_STR,
    MODEL_STR,
    OPTIMIZER_STR,
    WriteResult,
    _StorageInfo,
    _StoragePrefix,
)
from bytecheckpoint.engine.base_engine import BaseStoreEngine
from bytecheckpoint.planner.default_planner import DefaultSavePlanner
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger
from bytecheckpoint.utilities.serialization import _serialize_tensor

logger = get_bytecheckpoint_logger()

__all__ = ["StoreEngine"]


class StoreEngine(BaseStoreEngine):
    """
    Implementation of StoreEngine using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.
    """

    def __init__(
        self,
        single_file_per_rank: bool = True,
        sync_files: bool = True,
    ) -> None:
        super().__init__()
        self.single_file_per_rank = single_file_per_rank
        self.sync_files = sync_files

    def execute(self, ckpt_name: str, framework_name: str, suffix: Optional[str], *args, **kwargs) -> List[Future[Any]]:
        """
        Execute the appropriate storage method based on the checkpoint name.

        Args:
            ckpt_name (str): The name of the checkpoint, which determines the storage method.
            framework_name (str): The name of the framework.
            suffix (Optional[str]): An optional suffix for the resource key name.
            *args: Additional positional arguments to be passed to the storage method.
            **kwargs: Additional keyword arguments to be passed to the storage method.

        Returns:
            List[Future[Any]]: A list of futures representing the asynchronous storage operations.

        Raises:
            ValueError: If the provided `ckpt_name` is not supported.
        """
        # Select corresponding functions for execution.
        if ckpt_name in {MODEL_STR, OPTIMIZER_STR}:
            return self.store_model_optim_state(ckpt_name, framework_name, suffix, *args, **kwargs)
        elif ckpt_name in {LOADER_CKPT_STR, EXTRA_STATE_STR}:
            return self.store_extra_state(ckpt_name, framework_name, suffix, *args, **kwargs)
        else:
            raise ValueError(f"Do not support {ckpt_name} checkpoint type!")

    """
    Writing Methods.
    """

    def prepare_model_optim_state(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        tasks: List[Tuple[Path, str, List[WriteItem]]],
        planner: DefaultSavePlanner,
    ):
        """
        First stage of saving, Perform Copy data to CPU (D2H).

        Args:
            ckpt_name (str): Current checkpoint name.
            framework_name (str): Current framework name.
            suffix (Optional[str]): Suffix to resource key name.
            tasks (List[Tuple[Path, str, List[WriteItem]]]): Partitioned tasks for workers to conduct serialization and the actual saving.
            planner (DefaultSavePlanner): Save planner used to resolve the bytes and tensor data.

        Returns:
            Tuple[List[List[Tuple[io.BytesIO, WriteItem]]], List[List[Tuple[torch.Tensor, WriteItem]]], List[Tuple[Path, str]]]:
                A tuple containing three lists:
                1. byte_data_item_writes: List of byte data items to be written.
                2. tensor_data_item_writes: List of tensor data items to be written.
                3. file_path_names: List of file paths and names for the data items.

        NOTE:
            - Currently, we do D2H synchronously.
            - This function should be invoked after calling `self.sync_io_futures()`.
        """

        # prepare return results.
        byte_data_item_writes: List[List[Tuple[io.BytesIO, WriteItem]]] = []
        tensor_data_item_writes: List[List[Tuple[torch.Tensor, WriteItem]]] = []
        file_path_names: List[Tuple[Path, str]] = []

        # Prepare resources.
        pinned_memory_pool = self.get_mem_pool(ckpt_name, framework_name, suffix)

        # Perform D2H.
        pinned_tensor_list: List[List[torch.Tensor]] = []
        d2h_dump_start = time.time()
        for task in tasks:
            file_path, file_name, write_items = task
            byte_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]
            tensor_w = [wi for wi in write_items if wi.type != WriteItemType.BYTE_IO]
            byte_data_item = [(planner.resolve_data(wi), wi) for wi in byte_w]
            tensor_data_item = []
            # Copy to pinned CPU memory pool.
            pinned_tensors = []
            for item in tensor_w:
                tensor = planner.resolve_data(item).detach()
                # If a tensor's nbytes is not equal to underly storage nbytes (when slicing or viewing),
                # clone tensor to avoid store the underly tensor, leading to write or read amplification.
                if tensor.nbytes != tensor.untyped_storage().nbytes():
                    tensor = tensor.clone()
                assert tensor.untyped_storage().nbytes() == tensor.nbytes
                # Don't copy CPU tensor to pinned memory
                if BYTECHECKPOINT_GLOBAL_CONFIG.enable_pinned_memory_d2h:
                    tensor = pinned_memory_pool.copy_gpu_tensor_to_cpu_pinned_mem_pool(tensor, non_blocking=False)
                    pinned_tensors.append(tensor)
                else:
                    if tensor.device.type == "cpu":
                        tensor = tensor.clone()
                    else:
                        tensor = tensor.cpu()
                tensor_data_item.append((tensor, item))

            byte_data_item_writes.append(byte_data_item)
            tensor_data_item_writes.append(tensor_data_item)
            file_path_names.append((file_path, file_name))
            pinned_tensor_list.append(pinned_tensors)

        d2h_dump_time = time.time() - d2h_dump_start
        logger.debug("End waiting for D2H copy. Time cost: %s s", d2h_dump_time)
        # Record pinned tensors.
        self.set_mem_usage(ckpt_name, framework_name, suffix, pinned_tensor_list)
        return byte_data_item_writes, tensor_data_item_writes, file_path_names

    def store_model_optim_state(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        plan: SavePlan,
        planner: DefaultSavePlanner,
        async_io: bool = False,
    ) -> List[Future[List[WriteResult]]]:
        """
        Store the model and optimizer states based on the provided save plan.

        This method orchestrates the process of storing the model and optimizer states to disk.
        It first synchronizes any ongoing I/O operations from the last store event, then generates
        save tasks according to the number of I/O workers. The method then prepares the storage
        items and submits the tasks to the I/O workers for execution.

        Args:
            ckpt_name (str): The name of the checkpoint.
            framework_name (str): The name of the framework.
            suffix (Optional[str]): An optional suffix for the resource key name.
            plan (SavePlan): The save plan containing the items to be stored.
            planner (DefaultSavePlanner): The save planner used to resolve the bytes and tensor data.
            async_io (bool, optional): Whether to perform I/O operations asynchronously. Defaults to False.

        Returns:
            List[Future[List[WriteResult]]]: A list of futures representing the asynchronous storage operations.

        Raises:
            RuntimeError: If an error occurs during the storage process.
        """
        # Get the worker count and io worker.
        executor = self.get_io_workers(ckpt_name, framework_name, suffix)
        store_worker_count = self.get_io_store_worker_count(ckpt_name, framework_name, suffix)
        # Get checkpoint path and P2P tensors information.
        ckpt_path = self.get_path(ckpt_name, framework_name, suffix)

        # Sync the io futures of the last store event.
        logger.debug("Start waiting for last store events.")
        last_store_start_time = time.time()
        self.sync_io_futures(ckpt_name, framework_name, suffix)
        last_store_time = time.time() - last_store_start_time
        logger.debug("Finish waiting for last store events. Time cost: %s s", last_store_time)

        # Generate save tasks according to the I/O worker count.
        storage_plan: _StoragePrefix = plan.storage_data
        file_count = 0

        def gen_file():
            nonlocal file_count
            file_name = f"{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
            file_count += 1
            return file_name

        tasks: List[Tuple[Path, str, List[WriteItem]]] = []
        # Generate K tasks where K is the number of store_worker_count.
        if self.single_file_per_rank:
            for bucket in _split_by_size_and_type(store_worker_count, plan.items):
                file_name = gen_file()
                tasks.append((ckpt_path / file_name, file_name, bucket))
        # Generate K tasks where K is the number of write items.
        else:
            for item in plan.items:
                file_name = gen_file()
                tasks.append((ckpt_path / file_name, file_name, [item]))
        logger.debug("Rank %s writes its checkpoint into %s files", dist.get_rank(), len(tasks))

        # Prepare storage items.
        byte_data_item_writes, tensor_data_item_writes, file_path_names = self.prepare_model_optim_state(
            ckpt_name,
            framework_name,
            suffix,
            tasks,
            planner,
        )

        # Submit the tasks.
        store_futures = []
        for byte_data_item, tensor_data_item, file_path_name in zip(
            byte_data_item_writes, tensor_data_item_writes, file_path_names
        ):
            file_path, storage_key = file_path_name
            worker_args = (file_path, storage_key, byte_data_item, tensor_data_item, self.sync_files)
            curr_future = executor.submit(_write_files_per_proc_pipe, *worker_args)
            store_futures.append(curr_future)

        # Record current futures.
        self.set_io_futures(ckpt_name, framework_name, suffix, new_futures=store_futures)

        if not async_io:
            logger.debug("Start waiting for storing futures (serialization + save)")
            future_wait_start = time.time()
            self.sync_io_futures(ckpt_name, framework_name, suffix)
            future_wait_time = time.time() - future_wait_start
            logger.debug("End waiting for storing futures. Time cost: %s s", future_wait_time)
        return store_futures

    def store_extra_state(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        file_name: str,
        state_dict,
        async_io: bool = False,
    ) -> List[Future[Any]]:
        """
        Store the extra state dictionary to a file using I/O workers.

        This method synchronizes any ongoing I/O operations from the last store event,
        then submits a task to the I/O workers to write the provided state dictionary
        to a file. It can operate either synchronously or asynchronously based on the
        `async_io` parameter.

        Args:
            ckpt_name (str): The name of the checkpoint.
            framework_name (str): The name of the framework.
            suffix (Optional[str]): An optional suffix for the resource key name.
            file_name (str): The name of the file to store the state dictionary.
            state_dict: The state dictionary to be stored.
            async_io (bool, optional): Whether to perform I/O operations asynchronously. Defaults to False.

        Returns:
            List[Future[Any]]: A list of futures representing the asynchronous storage operations.
        """
        # Get I/O workers.
        executor = self.get_io_workers(ckpt_name, framework_name, suffix)

        # Get path.
        ckpt_path = self.get_path(ckpt_name, framework_name, suffix)
        file_path = ckpt_path / file_name

        # Sync the io futures of the last store event.
        logger.debug("Start waiting for last store events.")
        last_store_start_time = time.time()
        self.sync_io_futures(ckpt_name, framework_name, suffix)
        last_store_time = time.time() - last_store_start_time
        logger.debug("Finish waiting for last store events. Time cost: %s s", last_store_time)

        store_futures = []
        worker_args = (file_path, state_dict)
        store_futures.append(executor.submit(_write_extra_file, *worker_args))

        # Record current futures.
        self.set_io_futures(ckpt_name, framework_name, suffix, new_futures=store_futures)

        if not async_io:
            logger.debug("Start waiting for storing extra state_dict futures (serialization + save)")
            future_wait_start = time.time()
            self.sync_io_futures(ckpt_name, framework_name, suffix)
            future_wait_time = time.time() - future_wait_start
            logger.debug("End waiting for storing extra state_dict futures. Time cost: %s s", future_wait_time)
        return store_futures


"""
Helper methods.
"""


def _write_files_per_proc_pipe(
    file_path: Path,
    storage_key: str,
    byte_data_item: List[Tuple[io.BytesIO, WriteItem]],
    tensor_data_item: List[Tuple[torch.Tensor, WriteItem]],
    use_fsync: bool,
) -> List[WriteResult]:
    """
    Write byte and tensor data to a local file.

    This function is responsible for writing byte and tensor data to a specified file.
    It first writes the byte data directly to the file and then serializes and writes the tensor data.
    If the `use_fsync` parameter is set to True, it ensures that all data is written to the disk.

    Args:
        file_path (Path): The path to the file where the data will be written.
        storage_key (str): The storage key associated with the data.
        byte_data_item (List[Tuple[io.BytesIO, WriteItem]]): A list of byte data items to be written.
        tensor_data_item (List[Tuple[torch.Tensor, WriteItem]]): A list of tensor data items to be written.
        use_fsync (bool): Whether to use `os.fsync` to ensure data is written to disk.

    Returns:
        List[WriteResult]: A list of write results for each data item written to the file.
    """
    write_results = []
    try:
        with open(file_path, "wb") as stream:
            # For byte data, directly write byte data.
            for write_data, write_item in byte_data_item:
                content = write_data.getbuffer()
                result = _write_bytes_to_file(stream, content, write_item, storage_key)
                write_results.append(result)
            # For tensor data, perform serialization in process then do saving in the threadpool.
            for write_data, write_item in tensor_data_item:
                metadata, storagedata = _serialize_tensor(write_data)
                result = _write_tensor_to_file(stream, metadata, storagedata, write_item, storage_key)
                write_results.append(result)
            if use_fsync:
                os.fsync(stream.fileno())
        return write_results
    except Exception as e:
        logger.error(
            "[rank-%s] Error occurred when storing to local ckpt file: %s, exception: %s, this exception will be handled and "
            "will not interrupt the training process",
            dist.get_rank(),
            file_path,
            e,
        )
        raise RuntimeError(e)


def _write_extra_file(file_path: Path, state_dict) -> bool:
    """
    Write the extra state dictionary to a local file.

    This function is responsible for writing the provided state dictionary to a specified file.
    It uses `torch.save` with `pickle_protocol=4` to avoid issues when saving checkpoints larger than 4GB.
    If an exception occurs during the writing process, it logs the error and re-raises it as a `RuntimeError`.

    Args:
        file_path (Path): The path to the file where the state dictionary will be written.
        state_dict: The state dictionary to be written to the file.

    Returns:
        bool: True if the state dictionary was successfully written to the file, otherwise raises an exception.
    """
    try:
        # use pickle_protocol=4 to avoid the problem that ckpt cannot be saved if it exceeds 4GB
        torch.save(state_dict, file_path, pickle_protocol=4)
        return True
    except Exception as e:
        logger.error(
            "[rank-%s] Error occurred when storing to local ckpt file: %s, exception: %s, this exception will be handled and "
            "will not interrupt the training process",
            dist.get_rank(),
            file_path,
            e,
        )
        raise RuntimeError(e)


def _result_from_write_item(
    item: WriteItem,
    size_in_bytes: int,
    byte_metadata: _StorageInfo,
    tensor_metadata: bytes = None,
) -> WriteResult:
    return WriteResult(
        index=item.index,
        size_in_bytes=size_in_bytes,
        byte_metadata=byte_metadata,
        tensor_metadata=tensor_metadata,
    )


def _item_size(item: WriteItem) -> int:
    size = 1
    assert item.tensor_data is not None
    # Can't use math.prod as PT needs to support older python.
    for s in item.tensor_data.size:
        size *= s

    dtype = item.tensor_data.properties.dtype
    return size * torch._utils._element_size(dtype)


def _split_by_size_and_type(bins, items: List[WriteItem]) -> List[List[WriteItem]]:
    """
    Split a list of WriteItem objects into multiple buckets based on their type and size.

    This function takes a list of WriteItem objects and distributes them across a specified number of bins.
    It first separates the items into two groups: byte data and tensor data. The byte data items are evenly
    distributed among the bins, while the tensor data items are sorted by size and placed in the bin with
    the smallest total size.

    Args:
        bins (int): The number of bins to split the items into.
        items (List[WriteItem]): A list of WriteItem objects to be split.

    Returns:
        List[List[WriteItem]]: A list of lists, where each inner list represents a bin containing a subset of WriteItem objects.
    """
    if bins == 1:
        return [items]

    bytes_w = [wi for wi in items if wi.type == WriteItemType.BYTE_IO]
    tensor_w = [wi for wi in items if wi.type != WriteItemType.BYTE_IO]

    buckets: List[List[WriteItem]] = [[] for _ in range(bins)]
    bucket_sizes = [0 for _ in range(bins)]

    tensor_w.sort(key=_item_size, reverse=True)

    for i, wi in enumerate(bytes_w):
        buckets[i % bins].append(wi)

    for wi in tensor_w:
        # TODO replace with headq.
        idx = min(enumerate(bucket_sizes), key=lambda x: x[1])[0]
        buckets[idx].append(wi)
        bucket_sizes[idx] += _item_size(wi)

    return buckets


def _write_bytes_to_file(stream, content: bytes, write_item: WriteItem, storage_key: str) -> WriteResult:
    offset = stream.tell()
    stream.write(content)
    length = stream.tell() - offset
    return _result_from_write_item(write_item, length, _StorageInfo(storage_key, offset, length))


def _write_tensor_to_file(
    stream,
    metadata: bytes,
    storagedata: bytes,
    write_item: WriteItem,
    storage_key: str,
) -> WriteResult:
    offset = stream.tell()
    stream.write(storagedata)
    length = stream.tell() - offset
    return _result_from_write_item(write_item, length, _StorageInfo(storage_key, offset, length), metadata)
