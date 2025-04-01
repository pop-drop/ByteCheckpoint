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
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from threading import Event
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from bytecheckpoint import BYTECHECKPOINT_GLOBAL_CONFIG
from bytecheckpoint.checkpointer.meta_type import (
    SUPPORTED_CHECKPOINT_TYPES,
    SUPPORTED_FRAMEWORK_TYPES,
    SUPPORTED_ROLE_SUFFIX_TYPES,
)
from bytecheckpoint.distributed.rpc_context import get_stub
from bytecheckpoint.io import bfile
from bytecheckpoint.utilities.ckpt_format.common_utils import get_checkpoint_tracker_filename
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger
from bytecheckpoint.utilities.server import server_lib
from bytecheckpoint.utilities.sync_queue import SynchronizedQueue

logger = get_bytecheckpoint_logger()

# Global exit process flag.
_process_should_exit = Event()


# Background thread fetches the persisting functions and executes them until the process exits.
def _bg_thread_func(sync_queue: SynchronizedQueue):
    global _process_should_exit
    while True:
        try:
            func, eof = sync_queue.get()
            if eof:
                return
            func()
        except Exception as e:
            import traceback

            # TODO: set the process_exit flag.
            logger.error("Get exception in _bg_thread_func %s", e)
            logger.error(traceback.format_exc())
            _process_should_exit.set()
            break
        finally:
            sync_queue.task_done()


class CKPTCounter:
    def __init__(self, _N: int, _async: bool):
        assert _N > 0, "_N must be greater than 0"
        self._N: int = _N
        self._async: bool = _async
        self.counter = 0
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def increment(self, integrity_func: Callable = None, *args, **kwargs):
        """
        Increment the counter and perform integrity checks if necessary.

        This method increments the internal counter and checks if it has reached the specified limit _N.
        If the limit is reached, it optionally calls the provided integrity function and resets the counter.
        If the operation is asynchronous (self._async is True), it also notifies all waiting threads.

        Args:
            integrity_func (Callable, optional): A function to be called when the counter reaches _N. Defaults to None.
            *args: Variable length argument list to be passed to the integrity function.
            **kwargs: Arbitrary keyword arguments to be passed to the integrity function.

        Raises:
            AssertionError: If _N or _async is not set before calling this method.
        """
        assert self._N is not None, "_N must be set before increment"
        assert self._async is not None, "_async must be set before increment"
        with self.condition:
            self.counter += 1
            if self.counter == self._N:
                try:
                    if integrity_func:
                        integrity_func(*args, **kwargs)
                finally:
                    self.counter = 0
                    # if write with background threads (self._async is True), intergity-checking thread notifies all waiting threads.
                    if self._async:
                        self.condition.notify_all()
            else:
                # If write with background threads (self._async is True), current thread for integrity-checking thread's notification.
                if self._async:
                    self.condition.wait()


class _BaseStorage(ABC):
    """
    The Storage class persist checkpoints in different storage backends.
    """

    @abstractmethod
    def register_resources(self, ckpt_name: str, framework_name: str, suffix: Optional[str]) -> bool:
        pass

    @abstractmethod
    def cleanup_resources(self):
        pass

    @abstractmethod
    def run(*args, **kwargs):
        pass

    @abstractmethod
    def execute(self, ckpt_name: str, framework_name: str, suffix: Optional[str], *args, **kwargs) -> None:
        pass


class BaseStorageWriter(_BaseStorage):
    """
    Basic implementation of the StoreWriter class.
    """

    def __init__(self):
        # Prepare sync queues and background threads for supported checkpoint types.
        self._sync_queues: Dict[str, Optional[SynchronizedQueue]] = {}
        self._bg_threads: Dict[str, Optional[threading.Thread]] = {}
        for framework_name, ckpt_name, suffix in itertools.product(
            SUPPORTED_FRAMEWORK_TYPES, SUPPORTED_CHECKPOINT_TYPES, SUPPORTED_ROLE_SUFFIX_TYPES
        ):
            tmp_resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
            self._sync_queues[tmp_resource_key_name] = None
            self._bg_threads[tmp_resource_key_name] = None

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

    def register_resources(self, ckpt_name: str, framework_name: str, suffix: Optional[str]) -> bool:
        # Make assersion and generate resource_key_name.
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        try:
            # Register the sync queue.
            if resource_key_name in self._sync_queues and self._sync_queues.get(resource_key_name) is None:
                self._sync_queues[resource_key_name] = SynchronizedQueue()
            if resource_key_name in self._bg_threads and self._bg_threads.get(resource_key_name) is None:
                bg_thread = threading.Thread(
                    target=_bg_thread_func, args=(self._sync_queues[resource_key_name],), daemon=True
                )
                bg_thread.start()
                self._bg_threads[resource_key_name] = bg_thread
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
        for framework_name, ckpt_name, suffix in itertools.product(
            SUPPORTED_FRAMEWORK_TYPES, SUPPORTED_CHECKPOINT_TYPES, SUPPORTED_ROLE_SUFFIX_TYPES
        ):
            try:
                resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
                # Cleanup the sync queue.
                if resource_key_name in self._sync_queues and self._sync_queues.get(resource_key_name) is not None:
                    sync_queue = self._sync_queues[resource_key_name]
                    sync_queue.put((None, True))
                if resource_key_name in self._bg_threads and self._bg_threads.get(resource_key_name) is not None:
                    bg_thread = self._bg_threads[resource_key_name]
                    bg_thread.join(timeout=BYTECHECKPOINT_GLOBAL_CONFIG.write_ckpt_timeout)
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

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def _execute(self, ckpt_name: str, framework_name: str, suffix: Optional[str], *args, **kwargs) -> None:
        raise NotImplementedError

    def execute(self, ckpt_name: str, framework_name: str, suffix: Optional[str], *args, **kwargs) -> None:
        global _process_should_exit
        if _process_should_exit.is_set():
            sys.exit(1)
        return self._execute(ckpt_name, framework_name, suffix, *args, **kwargs)

    def wait_for_local_write_futures(self, *args, **kwargs):
        raise NotImplementedError

    def _handle_failure(self, *args, **kwargs):
        raise NotImplementedError()

    def _write_checkpoint_tracker(
        self,
        save_ckpt_start_time,
        rank: int,
        global_steps: Optional[int] = None,
        root_path: Optional[str] = None,
        callback: Optional[callable] = None,
    ):
        """Writes the checkpoint tracker file and executes the callback if provided."""
        if root_path is not None and global_steps is not None:
            tracker_filename = get_checkpoint_tracker_filename(root_path)
            bfile.atomic_write(
                tracker_filename,
                str(global_steps).encode(),
                skip_encryption=True,
            )
        save_ckpt_cost_time = time.time() - save_ckpt_start_time
        logger.info(
            "Rank=%s Successfully write checkpoint files to path %s. Total time: %s s",
            rank,
            root_path,
            save_ckpt_cost_time,
        )
        if callback:
            callback()

    def write_tracker(self, *args, **kwargs):
        raise NotImplementedError()

    def is_coordinator(self, rank: int):
        return rank == 0

    def server_lib_gather_results(self, rank, tag, timeout, global_failure_count):
        all_gather_start_time = time.time()
        try:
            results = server_lib.gather(
                get_stub(),
                0,
                rank,
                global_failure_count,
                tag=tag,
                timeout=timeout,
            )
        except Exception as e:
            # log grpc server status before raising exception
            self.get_server_status_by_tag(tag)
            raise RuntimeError(
                "Failed to gather checkpoint write / upload status with tag %s from all ranks via server_lib, error: %s",
                tag,
                e,
            )

        all_gather_cost_time = time.time() - all_gather_start_time
        logger.info(
            "Rank=%s Server_lib finished gathering results. Time cost: %s s",
            rank,
            all_gather_cost_time,
        )
        return results

    @staticmethod
    def get_server_status_by_tag(tag: str = None):
        """
        Get the status of the server.

        Args:
            tag (str, optional): The tag to query the barrier status. Defaults to None.
        """
        try:
            assert tag is not None, "tag is required"
            resp = server_lib.get_server_status(get_stub(), timeout=10)
            world_size = resp["world_size"]
            tag_items = resp["gather_dict"]
            if tag in tag_items:
                item = tag_items[tag]
                all_ranks = set(range(world_size))
                item_ranks = item["ranks"]
                item_contents = item["contents"]
                logger.info(
                    "Get tag status %s (world_size %s) from gRPC server: %s reported ranks, with contents %s",
                    tag,
                    world_size,
                    len(item_ranks),
                    item_contents,
                )
                missing_ranks = all_ranks - item_ranks
                if len(missing_ranks) > 0:
                    logger.warning(
                        "Please check the log in missed ranks %s. It is likely that checkpoint upload to FS failed.",
                        missing_ranks,
                    )
        except Exception as e:
            logger.warning("Failed to get gRPC server status, warning: %s", e)

    """
    Setter and Getter methods.
    """

    def get_sync_queue(self, ckpt_name: str, framework_name: str, suffix: Optional[str]) -> SynchronizedQueue:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._sync_queues, f"resource key {resource_key_name} is not supported!"
        futures = self._sync_queues.get(resource_key_name)
        assert futures, f"sync queue for {resource_key_name} is None, set the sync queue first!"
        return futures


class BaseStorageReader(_BaseStorage):
    """
    Basic implementation of the StoreReader class.
    """

    def __init__(self):
        # Prepare I/O workers and futures for supported checkpoint types.
        self._io_workers: Dict[str, Optional[Union[ProcessPoolExecutor, ThreadPoolExecutor]]] = {}
        self._io_futures: Dict[str, Optional[List[Future[Tuple[str, str]]]]] = {}
        # Prepare paths for supported checkpoint types.
        self._paths: Dict[str, Union[str, os.PathLike]] = {}
        self._temp_load_paths: Dict[str, Union[str, os.PathLike]] = {}
        for framework_name, ckpt_name, suffix in itertools.product(
            SUPPORTED_FRAMEWORK_TYPES, SUPPORTED_CHECKPOINT_TYPES, SUPPORTED_ROLE_SUFFIX_TYPES
        ):
            tmp_resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
            self._io_workers[tmp_resource_key_name] = None
            self._io_futures[tmp_resource_key_name] = None
            self._paths[tmp_resource_key_name] = None
            self._temp_load_paths[tmp_resource_key_name] = None

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
        # Make assersion and generate resource_key_name.
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        try:
            # Register the I/O workers.
            if resource_key_name in self._io_workers and self._io_workers.get(resource_key_name) is None:
                self._io_workers[resource_key_name] = ThreadPoolExecutor(
                    max_workers=BYTECHECKPOINT_GLOBAL_CONFIG.load_worker_count
                )

            # Register the checkpoint path.
            self.set_path(ckpt_name, framework_name, suffix, ckpt_path)
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

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def execute(self, ckpt_name: str, framework_name: str, suffix: Optional[str], *args, **kwargs) -> None:
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
        # Set path.
        self._paths[resource_key_name] = ckpt_path

    def get_path(self, ckpt_name: str, framework_name: str, suffix: Optional[str]) -> Union[str, os.PathLike]:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._paths, f"resource key {resource_key_name} is not supported!"
        path = self._paths.get(resource_key_name)
        assert path, f"Checkpoint path for {resource_key_name} is None, set path first!"
        return path

    def get_temp_load_path(self, ckpt_name: str, framework_name: str, suffix: Optional[str]) -> Union[str, os.PathLike]:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._paths, f"resource key {resource_key_name} is not supported!"
        path = self._temp_load_paths.get(resource_key_name)
        assert path, f"Checkpoint path for {resource_key_name} is None, set path first!"
        return path

    def set_io_futures(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        new_futures: Future[Any],
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

    def get_io_workers(
        self, ckpt_name: str, framework_name: str, suffix: Optional[str]
    ) -> Union[ProcessPoolExecutor, ThreadPoolExecutor]:
        resource_key_name = self._generate_resource_key_name(ckpt_name, framework_name, suffix)
        assert resource_key_name in self._io_workers, f" resource key {resource_key_name} is not supported!"
        executor = self._io_workers.get(resource_key_name)
        assert executor is not None, f"I/O workers for {resource_key_name} are not set yet, initialize them first!"
        return executor
