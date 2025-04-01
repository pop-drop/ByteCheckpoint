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

from concurrent.futures import Future
from typing import List, Optional, Tuple

from bytecheckpoint.storage._storage import _local_storage_reader
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()


def read_from_store(
    ckpt_name: str,
    framework_name: str,
    resource_key_name_suffix: Optional[str],
    relative_paths: List[str],
    path: str,
    async_io: bool = False,
) -> List[Future[Tuple[str, str]]]:
    """
    Read data from the storage.

    Args:
        ckpt_name (str): The name of the checkpoint.
        framework_name (str): The name of the framework.
        resource_key_name_suffix (Optional[str]): The suffix of the resource key name.
        relative_paths (List[str]): The list of relative paths.
        path (str): The path to the storage.
        async_io (bool, optional): Whether to use asynchronous I/O. Defaults to False.

    Returns:
        List[Future[Tuple[str, str]]]: A list of futures that will return a tuple of strings.
    """
    # If the path is a local / NFS path and need to write tracker.
    # we start function waiting for local futures and write tracker if global_steps exists.

    # Register the resources lazily.
    assert _local_storage_reader.register_resources(
        ckpt_name,
        framework_name,
        resource_key_name_suffix,
        path,
    )
    # Execution.
    return _local_storage_reader.execute(
        ckpt_name,
        framework_name,
        resource_key_name_suffix,
        relative_paths,
        async_io,
    )
