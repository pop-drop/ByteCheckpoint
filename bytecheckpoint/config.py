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

from bytecheckpoint.checkpointer.meta_type import SUPPORTED_ROLE_SUFFIX_TYPES


def update_role_suffixes_from_env() -> None:
    """
    Update the set of supported role suffix types with values from the environment variable.

    This function reads the environment variable "BYTECHECKPOINT_ROLE_SUFFIX",
    splits it by commas, and adds each non - empty, stripped, and uppercased value
    to the set of supported role suffix types.

    If the environment variable is not set or is empty after stripping,
    the function returns without making any changes.
    """
    raw_value = os.getenv("BYTECHECKPOINT_ROLE_SUFFIX", "").strip()
    if not raw_value:
        return []
    # Read all configured role suffixes.
    role_suffixes = {role.strip().lower() for role in raw_value.split(",") if role.strip()}
    SUPPORTED_ROLE_SUFFIX_TYPES.update(role_suffixes)


class ByteCheckpointGlobalConfig:
    """
    A class records all configurations controlled by environment variables
    """

    def __init__(self):
        # Planning.
        self.enable_tree_topo = bool(int(os.getenv("BYTECHECKPOINT_ENABLE_TREE_TOPO", 0)))
        assert self.enable_tree_topo in [True, False], "BYTECHECKPOINT_ENABLE_TREE_TOPO must be 0 or 1"

        self.planner_lru_cache_capacity = int(os.getenv("BYTECHECKPOINT_PLANNER_CACHE_CAPACITY", 8))
        assert self.planner_lru_cache_capacity >= 1, (
            "BYTECHECKPOINT_PLANNER_CACHE_CAPACITY must be greater than or equal to 1"
        )

        # Store and write checkpoint.
        self.store_use_thread_io_worker = bool(int(os.getenv("BYTECHECKPOINT_STORE_USE_THREAD_IO_WORKER", 1)))

        self.store_worker_count = int(os.getenv("BYTECHECKPOINT_STORE_WORKER_COUNT", 1))
        assert self.store_worker_count >= 1, "BYTECHECKPOINT_STORE_WORKER_COUNT must be greater than or equal to 1"

        self.enable_pinned_memory_d2h = bool(int(os.getenv("BYTECHECKPOINT_ENABLE_PINNED_MEM_D2H", 1)))
        assert self.enable_pinned_memory_d2h in [True, False], "BYTECHECKPOINT_ENABLE_PINNED_MEM_D2H must be 0 or 1"

        self.write_ckpt_timeout = int(os.getenv("BYTECHECKPOINT_WRITE_CKPT_TIMEOUT", 1800))

        # Read and load checkpoint.
        self.load_worker_count = int(os.getenv("BYTECHECKPOINT_LOAD_WORKER_COUNT", 8))
        assert self.load_worker_count >= 1, "BYTECHECKPOINT_LOAD_WORKER_COUNT must be greater than or equal to 1"

        # Merge checkpoint.
        self.merge_num_io_worker = int(os.getenv("BYTECHECKPOINT_MERGE_NUM_IO_WORKER", 16))
        assert self.merge_num_io_worker >= 1, "BYTECHECKPOINT_MERGE_NUM_IO_WORKER must be greater than or equal to 1"

        # Add additional role suffixes in multi-role training scenarios.
        update_role_suffixes_from_env()


BYTECHECKPOINT_GLOBAL_CONFIG = ByteCheckpointGlobalConfig()
