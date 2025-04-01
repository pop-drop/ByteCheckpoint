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

import itertools
import os
from abc import ABC, abstractmethod
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.distributed.checkpoint.metadata import STORAGE_TYPES, MetadataIndex

from bytecheckpoint import BYTECHECKPOINT_GLOBAL_CONFIG
from bytecheckpoint.checkpointer.meta_type import (
    MODEL_STR,
    OPTIMIZER_STR,
    SUPPORTED_CHECKPOINT_TYPES,
    SUPPORTED_FRAMEWORK_TYPES,
    SUPPORTED_ROLE_SUFFIX_TYPES,
    Metadata,
    WriteResult,
    _StorageInfo,
)
from bytecheckpoint.planner.common import GLOBAL_PLAN_CACHE
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

from .memory_pool import PinnedMemoryPool

logger = get_bytecheckpoint_logger()

PINNED_MEMORY_SUPPORTED_CHECKPOINT_TYPES = {MODEL_STR, OPTIMIZER_STR}
METADATA_SUPPORTED_CHECKPOINT_TYPES = {MODEL_STR, OPTIMIZER_STR}


class _BaseEngine(ABC):
    """
    The Engine class recieves save/load plans from the planners, select storage backends, then execute the I/O tasks
    with registered resources.
    """

    @abstractmethod
    def register_resources(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        ckpt_path: Union[str, os.PathLike],
    ) -> bool:
        pass

    @abstractmethod
    def cleanup_resources(self):
        pass

    @abstractmethod
    def execute(self, ckpt_name: str, framework_name: str, suffix: Optional[str], *args, **kwargs) -> List[Future[Any]]:
        pass

    @abstractmethod
    def sync_io_futures(self, ckpt_name: str, framework_name: str, suffix: Optional[str]) -> bool:
        pass


class BaseStoreEngine(_BaseEngine):
    """
    Basic implementation of the WriteEngine class.
    """

    def __init__(self) -> None:
        # Prepare I/O workers and futures for supported checkpoint types.
        self._io_workers: Dict[str, Optional[Union[ProcessPoolExecutor, ThreadPoolExecutor]]] = {}
        self._io_futures: Dict[str, Optional[Future[List[WriteResult]]]] = {}
        # Prepare paths for supported checkpoint types.
        self._paths: Dict[str, Optional[Path]] = {}
        self._mem_pools: Dict[str, Optional[PinnedMemoryPool]] = {}
        self._mem_usage: Dict[str, Optional[List[List[torch.Tensor]]]] = {}

        for framework_name, ckpt_name, suffix in itertools.product(
            SUPPORTED_FRAMEWORK_TYPES,
            SUPPORTED_CHECKPOINT_TYPES,
            SUPPORTED_ROLE_SUFFIX_TYPES,
        ):
            tmp_resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)

            self._io_workers[tmp_resource_key_name] = None
            self._io_futures[tmp_resource_key_name] = None
            self._paths[tmp_resource_key_name] = None

        for framework_name, ckpt_name, suffix in itertools.product(
            SUPPORTED_FRAMEWORK_TYPES,
            PINNED_MEMORY_SUPPORTED_CHECKPOINT_TYPES,
            SUPPORTED_ROLE_SUFFIX_TYPES,
        ):
            tmp_resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)

            self._mem_pools[tmp_resource_key_name] = None
            self._mem_usage[tmp_resource_key_name] = None

    @staticmethod
    def _generate_resource_key_name(ckpt_name: str, framework_name: str, suffix: Optional[str]) -> str:
        assert ckpt_name in SUPPORTED_CHECKPOINT_TYPES, f"received ckpt type: {ckpt_name} is not supported"
        assert framework_name in SUPPORTED_FRAMEWORK_TYPES, (
            f"received framework type: {framework_name} is not supported"
        )
        resource_key_name = f"{framework_name}_{ckpt_name}"
        if suffix:
            assert suffix in SUPPORTED_ROLE_SUFFIX_TYPES, f"received role suffix type: {suffix} is not supported"
            resource_key_name = f"{resource_key_name}_{suffix}"
        return resource_key_name

    def register_resources(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        ckpt_path: Union[str, os.PathLike],
    ) -> bool:
        # Make assertion and generate resource_key_name.
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        try:
            # Register I/O workers.
            store_worker_count = BYTECHECKPOINT_GLOBAL_CONFIG.store_worker_count
            if resource_key_name in self._io_workers and self._io_workers.get(resource_key_name) is None:
                if BYTECHECKPOINT_GLOBAL_CONFIG.store_use_thread_io_worker:
                    self._io_workers[resource_key_name] = ThreadPoolExecutor(max_workers=store_worker_count)
                else:
                    self._io_workers[resource_key_name] = ProcessPoolExecutor(max_workers=store_worker_count)

            # Register the checkpoint path.
            self.set_path(ckpt_name, framework_name, suffix, ckpt_path)

            # Register pinned memory pool.
            if resource_key_name in self._mem_pools and self._mem_pools.get(resource_key_name) is None:
                self._mem_pools[resource_key_name] = PinnedMemoryPool()
        except Exception as e:
            logger.error(
                "Fail to register resources for ckpt type %s, framework %s, suffix %s, due to exception: %s",
                ckpt_name,
                framework_name,
                suffix,
                e,
            )
            return False
        return True

    def cleanup_resources(self):
        # Cleanup the global cache for save plan and metadata.
        GLOBAL_PLAN_CACHE.clear()
        # Cleanup I/O workers and pinned memory pool.
        for framework_name, ckpt_name, suffix in itertools.product(
            SUPPORTED_FRAMEWORK_TYPES, SUPPORTED_CHECKPOINT_TYPES, SUPPORTED_ROLE_SUFFIX_TYPES
        ):
            try:
                resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
                # Sync and cleanup I/O workers.
                if resource_key_name in self._io_futures and self._io_futures.get(resource_key_name) is not None:
                    futures = self._io_futures[resource_key_name]
                    for future in futures:
                        future.result()
                    self._io_futures[resource_key_name] = None
                if resource_key_name in self._io_workers and self._io_workers.get(resource_key_name) is not None:
                    self._io_workers[resource_key_name].shutdown()
                    self._io_workers[resource_key_name] = None
                # Cleanup pinned memory pool.
                if resource_key_name in self._mem_usage and self._mem_usage.get(resource_key_name) is not None:
                    pinned_tensor_list: List[List[torch.Tensor]] = self._mem_usage[resource_key_name]
                    for pinned_tensors in pinned_tensor_list:
                        for tensor in pinned_tensors:
                            self._mem_pools[resource_key_name].deallocate_cpu_tensor_in_pinned_mem_pool(tensor)
                    self._mem_usage[resource_key_name] = None
                if resource_key_name in self._mem_pools and self._mem_pools.get(resource_key_name) is not None:
                    self._mem_pools[resource_key_name].release_memory_pool()
                    self._mem_pools[resource_key_name] = None
            except Exception as e:
                logger.error(
                    "Fail to cleanup resources for ckpt type %s, framework %s, suffix %s, due to exception: %s",
                    ckpt_name,
                    framework_name,
                    suffix,
                    e,
                )
                raise RuntimeError(
                    "Fail to cleanup resources for ckpt type %s, framework %s, suffix %s, due to exception: %s",
                    ckpt_name,
                    framework_name,
                    suffix,
                    e,
                )

    def execute(self, ckpt_name: str, framework_name: str, suffix: Optional[str], *args, **kwargs) -> List[Future[Any]]:
        """
        Execute the I/O tasks with separated resources specified by given ckpt_name and framework_name.
        """
        raise NotImplementedError

    def sync_io_futures(self, ckpt_name: str, framework_name: str, suffix: Optional[str]):
        try:
            # Make assersion and generate resource_key_name.
            resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
            # Wait for the completions of recorded I/O futures.
            if resource_key_name in self._io_futures and self._io_futures.get(resource_key_name) is not None:
                futures = self._io_futures[resource_key_name]
                for future in futures:
                    future.result()
                self._io_futures[resource_key_name] = None
            # Recycle allocated pinned memory segments.
            if resource_key_name in self._mem_usage and self._mem_usage.get(resource_key_name) is not None:
                pinned_tensor_list: List[List[torch.Tensor]] = self._mem_usage[resource_key_name]
                for pinned_tensors in pinned_tensor_list:
                    for tensor in pinned_tensors:
                        self._mem_pools[resource_key_name].deallocate_cpu_tensor_in_pinned_mem_pool(tensor)
                self._mem_usage[resource_key_name] = None
        except Exception as e:
            logger.error(
                "Fail to sync I/O futures for ckpt type %s, framework %s, suffix %s, due to exception: %s",
                ckpt_name,
                framework_name,
                suffix,
                e,
            )
            raise RuntimeError(
                "Fail to sync I/O futures for ckpt type %s, framework %s, suffix %s, due to exception: %s",
                ckpt_name,
                framework_name,
                suffix,
                e,
            )

    """
    Setter and Getter methods.
    """

    def set_io_futures(
        self, ckpt_name: str, framework_name: str, suffix: Optional[str], new_futures: Future[Any]
    ) -> None:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._io_futures, f" resource key {resource_key_name} is not supported!"
        futures = self._io_futures.get(resource_key_name)
        assert futures is None, f"I/O futures for {resource_key_name} are not None, sync I/O opreations first!"
        self._io_futures[resource_key_name] = new_futures

    def get_io_futures(self, ckpt_name: str, framework_name: str, suffix: Optional[str]) -> Future[Any]:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._io_futures, f"resource key {resource_key_name} is not supported!"
        futures = self._io_futures.get(resource_key_name)
        assert futures, f"I/O futures for {resource_key_name} are None, set I/O futures first!"
        return futures

    def set_io_workers(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: str,
        new_executor: Union[ProcessPoolExecutor, ThreadPoolExecutor],
    ):
        raise NotImplementedError("set_io_workers is not implemented!")

    def get_io_workers(
        self, ckpt_name: str, framework_name: str, suffix: Optional[str]
    ) -> Union[ProcessPoolExecutor, ThreadPoolExecutor]:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._io_workers, f" resource key {resource_key_name} is not supported!"
        executor = self._io_workers.get(resource_key_name)
        assert executor is not None, f"I/O workers for {resource_key_name} are not set yet, initialize them first!"
        return executor

    def get_io_store_worker_count(self, ckpt_name: str, framework_name: str, suffix: Optional[str]) -> int:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._io_workers, f" resource key {resource_key_name} is not supported!"
        return self.get_io_workers(ckpt_name, framework_name, suffix)._max_workers

    def set_mem_usage(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        new_pinned_tensor_list: List[List[torch.Tensor]],
    ) -> None:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._mem_usage, f" resource key {resource_key_name} is not supported!"
        mem_usage = self._mem_usage.get(resource_key_name)
        assert mem_usage is None, (
            f"Pinned memory usage for {resource_key_name} is not None, recycle pinned tensors first!"
        )
        self._mem_usage[resource_key_name] = new_pinned_tensor_list

    def get_mem_usage(self, ckpt_name: str, framework_name: str, suffix: Optional[str]) -> List[List[torch.Tensor]]:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._mem_usage, f" resource key {resource_key_name} is not supported!"
        mem_usage = self._mem_usage.get(resource_key_name)
        assert mem_usage, f"Pinned memory usage for {resource_key_name} is None, record pinned tensors first!"
        return mem_usage

    def get_mem_pool(self, ckpt_name: str, framework_name: str, suffix: Optional[str]) -> PinnedMemoryPool:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._mem_pools, f" resource key {resource_key_name} is not supported!"
        mem_pool = self._mem_pools.get(resource_key_name)
        assert mem_pool is not None, f"Memory pool for {resource_key_name} is not set yet, initialize it first!"
        return mem_pool

    def set_path(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        ckpt_path: Union[str, os.PathLike],
    ) -> None:
        # Allow directly reset the path.
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._paths, f" resource key {resource_key_name} is not supported!"
        self._paths[resource_key_name] = Path(ckpt_path)
        self._paths[resource_key_name].mkdir(parents=True, exist_ok=True)

    def get_path(self, ckpt_name: str, framework_name: str, suffix: Optional[str]) -> Path:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._paths, f"resource key {resource_key_name} is not supported!"
        path = self._paths.get(resource_key_name)
        assert path, f"Checkpoint path for {resource_key_name} is None, set path first!"
        return path


class BaseLoadEngine(_BaseEngine):
    """
    Basic implementation of the ReadEngine class.
    """

    def __init__(self) -> None:
        # Prepare I/O workers and futures for supported checkpoint types.
        self._io_workers: Dict[str, Optional[Union[ProcessPoolExecutor, ThreadPoolExecutor]]] = {}
        self._io_futures: Dict[str, Optional[Future[List[WriteResult]]]] = {}
        # Prepare storage_data and all_tensor_metadata for supported checkpoint types.
        self._use_legacy: Dict[str, Optional[bool]] = {}
        self._state_dict_metadata: Dict[str, Optional[Dict[str, STORAGE_TYPES]]] = {}
        self._storage_data: Dict[str, Optional[Dict[MetadataIndex, _StorageInfo]]] = {}
        self._all_tensor_metadata: Dict[str, Optional[Dict[MetadataIndex, bytes]]] = {}

        for framework_name, ckpt_name, suffix in itertools.product(
            SUPPORTED_FRAMEWORK_TYPES,
            SUPPORTED_CHECKPOINT_TYPES,
            SUPPORTED_ROLE_SUFFIX_TYPES,
        ):
            tmp_resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)

            self._io_workers[tmp_resource_key_name] = None
            self._io_futures[tmp_resource_key_name] = None

        for framework_name, ckpt_name, suffix in itertools.product(
            SUPPORTED_FRAMEWORK_TYPES,
            METADATA_SUPPORTED_CHECKPOINT_TYPES,
            SUPPORTED_ROLE_SUFFIX_TYPES,
        ):
            tmp_resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)

            self._use_legacy[tmp_resource_key_name] = None
            self._state_dict_metadata[tmp_resource_key_name] = None
            self._storage_data[tmp_resource_key_name] = None
            self._all_tensor_metadata[tmp_resource_key_name] = None

    @staticmethod
    def _generate_resource_key_name(ckpt_name: str, framework_name: str, suffix: Optional[str]) -> str:
        assert ckpt_name in SUPPORTED_CHECKPOINT_TYPES, f"received ckpt type: {ckpt_name} is not supported"
        assert framework_name in SUPPORTED_FRAMEWORK_TYPES, (
            f"received framework type: {framework_name} is not supported"
        )
        resource_key_name = f"{framework_name}_{ckpt_name}"
        if suffix:
            assert suffix in SUPPORTED_ROLE_SUFFIX_TYPES, f"received role suffix type: {suffix} is not supported"
            resource_key_name = f"{resource_key_name}_{suffix}"
        return resource_key_name

    def register_resources(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
    ) -> bool:
        # Make assertion and generate resource_key_name.
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        try:
            # Register I/O workers.
            if resource_key_name in self._io_workers and self._io_workers.get(resource_key_name) is None:
                self._io_workers[resource_key_name] = ThreadPoolExecutor(
                    max_workers=BYTECHECKPOINT_GLOBAL_CONFIG.load_worker_count
                )

        except Exception as e:
            logger.error(
                "Fail to register resources for ckpt type %s, framework %s, suffix %s, due to exception: %s",
                ckpt_name,
                framework_name,
                suffix,
                e,
            )
            return False
        return True

    def cleanup_resources(self):
        # Cleanup the global cache for save plan and metadata.
        GLOBAL_PLAN_CACHE.clear()
        # Cleanup I/O workers and pinned memory pool.
        for framework_name, ckpt_name, suffix in itertools.product(
            SUPPORTED_FRAMEWORK_TYPES, SUPPORTED_CHECKPOINT_TYPES, SUPPORTED_ROLE_SUFFIX_TYPES
        ):
            try:
                resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
                # Sync and cleanup I/O workers.
                if resource_key_name in self._io_futures and self._io_futures.get(resource_key_name) is not None:
                    futures = self._io_futures[resource_key_name]
                    for future in futures:
                        future.result()
                    self._io_futures[resource_key_name] = None
                if resource_key_name in self._io_workers and self._io_workers.get(resource_key_name) is not None:
                    self._io_workers[resource_key_name].shutdown()
                    self._io_workers[resource_key_name] = None
            except Exception as e:
                logger.error(
                    "Fail to cleanup resources for ckpt type %s, framework %s, suffix %s, due to exception: %s",
                    ckpt_name,
                    framework_name,
                    suffix,
                    e,
                )
                raise RuntimeError(
                    "Fail to cleanup resources for ckpt type %s, framework %s, suffix %s, due to exception: %s",
                    ckpt_name,
                    framework_name,
                    suffix,
                    e,
                )

    def execute(self, ckpt_name: str, framework_name: str, suffix: Optional[str], *args, **kwargs) -> List[Future[Any]]:
        """
        Execute the I/O tasks with separated resources specified by given ckpt_name and framework_name.
        """
        raise NotImplementedError

    def sync_io_futures(self, ckpt_name: str, framework_name: str, suffix: Optional[str]):
        try:
            # Make assersion and generate resource_key_name.
            resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
            # Wait for the completions of recorded I/O futures.
            if resource_key_name in self._io_workers and self._io_futures.get(resource_key_name) is not None:
                futures = self._io_futures[resource_key_name]
                for future in futures:
                    future.result()
                self._io_futures[resource_key_name] = None
        except Exception as e:
            logger.error(
                "Fail to sync I/O futures for ckpt type %s, framework %s, suffix %s, due to exception: %s",
                ckpt_name,
                framework_name,
                suffix,
                e,
            )
            raise RuntimeError(
                "Fail to sync I/O futures for ckpt type %s, framework %s, suffix %s, due to exception: %s",
                ckpt_name,
                framework_name,
                suffix,
                e,
            )

    def set_storage_backend(self, *args, **kwargs) -> None:
        raise NotImplementedError

    """
    Setter and Getter methods.
    """

    def set_io_futures(
        self, ckpt_name: str, framework_name: str, suffix: Optional[str], new_futures: Future[Any]
    ) -> None:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._io_futures, f" resource key {resource_key_name} is not supported!"
        futures = self._io_futures.get(resource_key_name)
        assert futures is None, f"I/O futures for {resource_key_name} are not None, sync I/O opreations first!"
        self._io_futures[resource_key_name] = new_futures

    def get_io_futures(self, ckpt_name: str, framework_name: str, suffix: Optional[str]) -> Future[Any]:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._io_futures, f"resource key {resource_key_name} is not supported!"
        futures = self._io_futures.get(resource_key_name)
        assert futures, f"I/O futures for {resource_key_name} are None, set I/O futures first!"
        return futures

    def set_io_workers(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: str,
        new_executor: Union[ProcessPoolExecutor, ThreadPoolExecutor],
    ):
        raise NotImplementedError("set_io_workers is not implemented!")

    def get_io_workers(
        self, ckpt_name: str, framework_name: str, suffix: Optional[str]
    ) -> Union[ProcessPoolExecutor, ThreadPoolExecutor]:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._io_workers, f" resource key {resource_key_name} is not supported!"
        executor = self._io_workers.get(resource_key_name)
        assert executor is not None, f"I/O workers for {resource_key_name} are not set yet, initialize them first!"
        return executor

    def get_io_store_worker_count(self, ckpt_name: str, framework_name: str, suffix: Optional[str]) -> int:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._io_workers, f" resource key {resource_key_name} is not supported!"
        return self.get_io_workers(ckpt_name, framework_name, suffix)._max_workers

    def set_metadata(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        metadata: Metadata = None,
    ):
        # Define the loading mode.
        if metadata is None:
            return

        # Set metadata according to version.
        assert metadata.state_dict_metadata is not None
        assert metadata.storage_data is not None
        assert metadata.all_tensor_metadata is not None
        self.set_state_dict_metadata(ckpt_name, framework_name, suffix, metadata.state_dict_metadata)
        self.set_storage_data(ckpt_name, framework_name, suffix, metadata.storage_data)
        self.set_all_tensor_metadata(ckpt_name, framework_name, suffix, metadata.all_tensor_metadata)

    def set_state_dict_metadata(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        new_state_dict_metadata: Dict[str, STORAGE_TYPES],
    ) -> None:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._state_dict_metadata, f" resource key {resource_key_name} is not supported!"
        self._state_dict_metadata[resource_key_name] = new_state_dict_metadata

    def get_state_dict_metadata(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
    ) -> Dict[str, STORAGE_TYPES]:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._state_dict_metadata, f"resource key {resource_key_name} is not supported!"
        state_dict_metadata = self._state_dict_metadata.get(resource_key_name)
        return state_dict_metadata

    def set_storage_data(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        new_storage_data: Dict[MetadataIndex, _StorageInfo],
    ) -> None:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._storage_data, f" resource key {resource_key_name} is not supported!"
        self._storage_data[resource_key_name] = new_storage_data

    def get_storage_data(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
    ) -> Dict[MetadataIndex, _StorageInfo]:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._storage_data, f"resource key {resource_key_name} is not supported!"
        storage_data = self._storage_data.get(resource_key_name)
        return storage_data

    def set_all_tensor_metadata(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        new_all_tensor_metadata: Dict[MetadataIndex, bytes],
    ) -> None:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._all_tensor_metadata, f" resource key {resource_key_name} is not supported!"
        self._all_tensor_metadata[resource_key_name] = new_all_tensor_metadata

    def get_all_tensor_metadata(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
    ) -> Dict[MetadataIndex, bytes]:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._all_tensor_metadata, f"resource key {resource_key_name} is not supported!"
        all_tensor_metadata = self._all_tensor_metadata.get(resource_key_name)
        return all_tensor_metadata
