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

from concurrent.futures import Future
from typing import Any, Dict, List, Optional

import torch.distributed as dist

from bytecheckpoint.distributed.rpc_context import setup_rpc_service
from bytecheckpoint.storage._storage import CKPTCounter, _local_storage_writer
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()


def write_to_store(
    ckpt_name: str,
    framework_name: str,
    suffix: Optional[str],
    fast_saving: bool,
    ckpt_path: str,
    save_ckpt_start_time,
    local_write_futures: Dict[str, List[Future[Any]]],
    ckpt_counter: CKPTCounter,
    global_steps: Optional[int] = None,
    callback: Optional[callable] = None,
):
    """
    Write the checkpoint to the remote storage.

    Args:
        ckpt_name (str): The name of the checkpoint.
        framework_name (str): The name of the framework.
        suffix (Optional[str]): The suffix of the checkpoint.
        fast_saving (bool): Whether to save the checkpoint asynchronously.
        ckpt_path (str): The path to the checkpoint.
        save_ckpt_start_time: The start time of saving the checkpoint.
        local_write_futures (Dict[str, List[Future[Any]]]): A dictionary of local write futures.
        ckpt_counter (CKPTCounter): The checkpoint counter.
        global_steps (Optional[int], optional): The global steps. Defaults to None.
        callback (Optional[callable], optional): The callback function. Defaults to None.
    """
    # Ensure RPC service for gathering uploading results is called.
    setup_rpc_service()

    # Register the resources lazily.
    assert _local_storage_writer.register_resources(ckpt_name, framework_name, suffix)

    # Execution.
    _local_storage_writer.execute(
        ckpt_name,
        framework_name,
        suffix,
        fast_saving,
        save_ckpt_start_time,
        dist.get_rank(),
        local_write_futures,
        ckpt_counter,
        global_steps,
        ckpt_path,
        callback,
    )
