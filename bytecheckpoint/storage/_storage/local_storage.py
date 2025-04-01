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

import os
import time
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple, Union

from bytecheckpoint import BYTECHECKPOINT_GLOBAL_CONFIG
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

from .base_storage import BaseStorageReader, BaseStorageWriter, CKPTCounter

logger = get_bytecheckpoint_logger()


class LocalStorageWriter(BaseStorageWriter):
    """
    A class manages persisting checkpoints to NFS.
    """

    def __init__(self):
        super().__init__()

    def run(
        self,
        save_ckpt_start_time,
        rank: int,
        local_write_futures: Dict[str, List[Future[Any]]],
        ckpt_counter: CKPTCounter,
        global_steps: Optional[int] = None,
        root_path: Optional[str] = None,
        callback: Optional[callable] = None,
        barrier_rpc_tag_suffix: Optional[str] = None,
    ):
        """
        Runs the local write and tracker writing process.

        Args:
            save_ckpt_start_time: The start time of saving the checkpoint.
            rank (int): The rank of the current process.
            local_write_futures (Dict[str, List[Future[Any]]]): A dictionary of local write futures.
            ckpt_counter (CKPTCounter): The checkpoint counter.
            global_steps (Optional[int]): The global steps of the checkpoint. Defaults to None.
            root_path (Optional[str]): The root path of the checkpoint. Defaults to None.
            callback (Optional[callable]): The callback function. Defaults to None.
            barrier_rpc_tag_suffix (Optional[str]): The suffix of the barrier RPC tag. Defaults to None.
        """
        global_failure_count_list = [0]
        # Wait for local write futures to complete.
        self.wait_for_local_write_futures(
            rank,
            global_failure_count_list,
            local_write_futures,
        )
        # Increment the ckpt counter and write the tracker.
        ckpt_counter.increment(
            self.write_tracker,
            save_ckpt_start_time=save_ckpt_start_time,
            rank=rank,
            global_failure_count=global_failure_count_list[0],
            global_steps=global_steps,
            root_path=root_path,
            callback=callback,
            barrier_rpc_tag_suffix=barrier_rpc_tag_suffix,
        )

    def _execute(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        fast_saving: bool,
        save_ckpt_start_time,
        rank: int,
        local_write_futures: Dict[str, List[Future[Any]]],
        ckpt_counter: CKPTCounter,
        global_steps: Optional[int] = None,
        root_path: Optional[str] = None,
        callback: Optional[callable] = None,
    ) -> None:
        def persist_func():
            self.run(
                save_ckpt_start_time,
                rank=rank,
                local_write_futures=local_write_futures,
                ckpt_counter=ckpt_counter,
                global_steps=global_steps,
                root_path=root_path,
                callback=callback,
                barrier_rpc_tag_suffix=suffix,
            )

        if fast_saving:
            # Get the corresponding sync_queue according to the resource key.
            sync_queue = self.get_sync_queue(ckpt_name, framework_name, suffix)
            sync_queue.put((persist_func, False))
        else:
            persist_func()

    def wait_for_local_write_futures(
        self,
        rank: int,
        global_failure_count_list: List[int],
        local_write_futures: Dict[str, List[Future[Any]]],
    ):
        try:
            logger.info("Rank=%s Start waiting for local write futures", rank)
            local_write_start = time.time()
            for futures in local_write_futures.values():
                for future in futures:
                    future.result()
            local_write_end = time.time() - local_write_start
            logger.info(
                "Rank=%s Finished waiting for local write futures. Time cost: %s s",
                rank,
                local_write_end,
            )
        except Exception as e:
            logger.error("Rank=%s Failed to wait for local write, error: %s", rank, e)
            global_failure_count_list[0] += 1

    def write_tracker(
        self,
        save_ckpt_start_time,
        rank: int,
        global_failure_count: int,
        global_steps: Optional[int] = None,
        root_path: Optional[str] = None,
        callback: Optional[callable] = None,
        barrier_rpc_tag_suffix: str = None,
    ):
        """
        Gathers failure statuses from all ranks and writes the checkpoint tracker if successful.

        Args:
            save_ckpt_start_time: The start time of saving the checkpoint.
            rank (int): The rank of the current process.
            global_failure_count (int): The global failure count.
            global_steps (Optional[int]): The global steps of the checkpoint. Defaults to None.
            root_path (Optional[str]): The root path of the checkpoint. Defaults to None.
            callback (Optional[callable]): The callback function. Defaults to None.
            barrier_rpc_tag_suffix (str): The suffix of the barrier RPC tag. Defaults to None.
        """
        if global_steps is None:
            return
        results = self.server_lib_gather_results(
            rank,
            tag=f"LocalWriteCheckpointBarrierStep{global_steps}{barrier_rpc_tag_suffix}",
            timeout=BYTECHECKPOINT_GLOBAL_CONFIG.write_ckpt_timeout,
            global_failure_count=global_failure_count,
        )
        if self.is_coordinator(rank):
            aggregated_result = sum(results)
            if aggregated_result == 0:
                self._write_checkpoint_tracker(save_ckpt_start_time, rank, global_steps, root_path, callback)
            else:
                self._handle_failure(results)

        logger.info(
            "Rank %s waits for barrier global_step=%s root_path=%s",
            rank,
            global_steps,
            root_path,
        )

    def _handle_failure(self, results):
        """
        Handles the scenario when writing checkpoint files fails on some ranks.

        Args:
            results (List[int]): A list of results indicating the success or failure of writing checkpoint files on each rank.
                                 Each element in the list corresponds to a rank, where 0 indicates success and non-zero indicates failure.
        Raises:
            RuntimeError: If any rank fails to write checkpoint files, a RuntimeError is raised with information about the failed ranks.
        """
        failed_ranks = [str(i) for i, res in enumerate(results) if res != 0]
        logger.error(
            "Failed to write checkpoint files to local path. Failed ranks: %s",
            failed_ranks,
        )
        raise RuntimeError(
            "Failed to write checkpoint files to local path. Failed ranks: %s",
            failed_ranks,
        )


class LocalStorageReader(BaseStorageReader):
    """
    A class manages reading checkpoints from NFS.
    """

    def __init__(self):
        super().__init__()

    def run(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        relative_paths: List[Union[str, os.PathLike]],
        async_io: bool,
    ) -> List[Future[Tuple[str, str]]]:
        """
        Runs the process to read checkpoints from NFS.

        Args:
            ckpt_name (str): The name of the checkpoint.
            framework_name (str): The name of the framework.
            suffix (Optional[str]): The suffix of the checkpoint.
            relative_paths (List[Union[str, os.PathLike]]): A list of relative paths to the checkpoints.
            async_io (bool): Whether to use asynchronous I/O.

        Returns:
            List[Future[Tuple[str, str]]]: A list of futures that will resolve to tuples containing the relative path and the full path to the checkpoint.
        """
        # Sync last I/O futures.
        self.sync_io_futures(ckpt_name, framework_name, suffix)
        # Prepare ckpt_path and I/O workers.
        ckpt_path = self.get_path(ckpt_name, framework_name, suffix)
        executor = self.get_io_workers(ckpt_name, framework_name, suffix)
        # Prepare checkpoints.
        futures = []
        for relative_path in relative_paths:
            futures.append(executor.submit(prepare_local_checkpoint, ckpt_path, relative_path))
        # record the I/O futures.
        self.set_io_futures(ckpt_name, framework_name, suffix, futures)
        # Sync I/O futures.
        if not async_io:
            self.sync_io_futures(ckpt_name, framework_name, suffix)
        return futures

    def execute(
        self,
        ckpt_name: str,
        framework_name: str,
        suffix: Optional[str],
        *args,
        **kwargs,
    ) -> List[Future[Tuple[str, str]]]:
        return self.run(ckpt_name, framework_name, suffix, *args, **kwargs)


def prepare_local_checkpoint(ckpt_path: str, relative_path: str) -> Tuple[str, str]:
    return relative_path, os.path.join(ckpt_path, relative_path)
