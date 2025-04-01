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

# meta_type.py saves all constants and data types commonly used in bytecheckpoint.

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, TypeVar

from torch.distributed.checkpoint.metadata import STORAGE_TYPES, MetadataIndex
from typing_extensions import Protocol, runtime_checkable

STATE_DICT_TYPE = Dict[str, Any]

# Checkpoint state types.
MODEL_STR = "model"
OPTIMIZER_STR = "optimizer"
STATE_DICT_STR = "state_dict"
EXTRA_STATE_STR = "extra_state"
LOADER_CKPT_STR = "loader_ckpt"

FSDP_STR = "fsdp"
FSDP2_STR = "fsdp2"
DDP_STR = "ddp"
# RL roles related types.
ACTOR_STR = "actor"
CRITIC_STR = "critic"
REF_STR = "ref"
RM_STR = "rm"
DEFAULT_STR = "default"
NON_ROLE_STR = ""
SHM_PATH = "/dev/shm"
MAIN_MODEL = "main_model"
FSDP_FLAT_PARAM_TO_FQNS = "fsdp_flat_param_to_fqns"
FSDP_IS_PADDING_MASK = "fsdp_is_padding_mask"
FSDP_NUMELS_WITH_PADDING = "fsdp_numels_with_padding"
FSDP_SHAPES = "fsdp_shapes"
FSDP_FLAT_PARAM_META = "fsdp_flat_param_meta"
MODEL_DICT_KEYS = "model_dict_keys"

FSDP_OPTIMIZER_REQUIRES_GRAD_NUMEL = "fsdp_optimizer_requires_grad_numel"
# bytecheckpoint local temp dir name for uploading
job_launch_timestamp = f"_{os.getenv('JOB_LAUNCH_TIMESTAMP', '')}"
BYTECHECKPOINT_LOCAL_COMMON_TEMP_DIR_FOR_UPLOADING_PREFIX = "bytecheckpoint_local_temp_dir_for_uploading_"
BYTECHECKPOINT_LOCAL_TEMP_DIR_FOR_UPLOADING_PREFIX = (
    BYTECHECKPOINT_LOCAL_COMMON_TEMP_DIR_FOR_UPLOADING_PREFIX + job_launch_timestamp
)
BYTECHECKPOINT_LOCAL_COMMON_TEMP_DIR_FOR_DOWNLOADING = "bytecheckpoint_load_checkpoint_dir"
BYTECHECKPOINT_LOCAL_TEMP_DIR_FOR_DOWNLOADING = (
    BYTECHECKPOINT_LOCAL_COMMON_TEMP_DIR_FOR_DOWNLOADING + job_launch_timestamp
)

_DIRECTORY_FORMAT = "global_step_{}"
_EXTRA_STATE_FORMAT = "extra_state_rank_{}.pt"

# All current supported checkpoint types and frameworks.
SUPPORTED_CHECKPOINT_TYPES = {MODEL_STR, OPTIMIZER_STR, LOADER_CKPT_STR, EXTRA_STATE_STR}
SUPPORTED_FRAMEWORK_TYPES = {FSDP_STR, FSDP2_STR, DDP_STR}
SUPPORTED_ROLE_SUFFIX_TYPES = {NON_ROLE_STR, ACTOR_STR, CRITIC_STR, REF_STR, RM_STR, DEFAULT_STR}

# All current supported frameworks for merging checkpoint.
SUPPORTED_MERGING_FRAMEWORK_TYPES = {FSDP_STR, FSDP2_STR, DDP_STR}


class SupportedStrategy(Enum):
    FSDP = 1


@runtime_checkable
class Stateful(Protocol):
    def state_dict(self) -> Dict[str, Any]: ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...


T = TypeVar("T", bound=Stateful)
CheckpointState = Dict[str, T]


@dataclass
class _StorageInfo:
    """
    This is the per entry storage info
    """

    relative_path: str
    offset: int
    length: int


@dataclass
class _StoragePrefix:
    prefix: str


@dataclass(frozen=True)
class WriteResult:
    index: MetadataIndex

    size_in_bytes: int
    byte_metadata: _StorageInfo
    # NOTE: byte object does not have tensor_metadata field.
    tensor_metadata: bytes = None


@dataclass
class Metadata:
    # Keys are the same from the `state_dict` used.
    state_dict_metadata: Dict[str, STORAGE_TYPES]
    planner_data: Any = None
    # storage metadata.
    storage_data: Dict[MetadataIndex, _StorageInfo] = None
    # All tensor metadata
    all_tensor_metadata: Dict[MetadataIndex, Optional[bytes]] = None
    # Version information.
    bytecheckpoint_version: str = None
    # User Defined Dict
    # Users can save any necessary data they desire.
    user_defined_dict: Dict[str, Any] = field(default_factory=dict)


DEFAULT_SUFFIX = ".distcp"
