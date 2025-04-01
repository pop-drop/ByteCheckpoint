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

DDP_ENABLED = True
FSDP_ENABLED = True
FSDP2_ENABLED = True

from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()

from bytecheckpoint.checkpointer.ddp_checkpointer import DDPCheckpointer
from bytecheckpoint.checkpointer.fsdp_checkpointer import FSDPCheckpointer

try:
    from bytecheckpoint.checkpointer.fsdp2_checkpointer import FSDP2Checkpointer
except Exception as e:
    logger.warning("Failed to import FSDP2Checkpointer.")
    FSDP2_ENABLED = False
    pass

from bytecheckpoint.checkpointer.meta_type import SUPPORTED_FRAMEWORK_TYPES, CheckpointState


def load(
    ckpt_path: str,
    checkpoint_state: CheckpointState,
    framework: str,
    fast_loading: bool = False,
    strict: bool = True,
    role: str = None,
    *,
    load_decomposed_model_optimizer: bool = False,
):
    """
    Load API to load a checkpoint from a specified path.

    Args:
        ckpt_path (str): The path to the checkpoint file.
        checkpoint_state (CheckpointState): The state of the checkpoint to be loaded.
        framework (str): The framework to use for loading the checkpoint. Must be one of the supported frameworks.
        fast_loading (bool, optional): Whether to use parallel loading. Defaults to False.
        strict (bool, optional): Whether to enforce strict loading. Defaults to True.
        role (str, optional): The role of the model and optimizer to be laoded, e.g. 'actor' or 'critic' in RL. Defaults to None.

    Raises:
        ValueError: If the specified framework is not supported or not enabled.

    Returns:
        The result of the checkpoint loading operation.
    """
    if framework not in SUPPORTED_FRAMEWORK_TYPES:
        raise ValueError(
            f"Framework {framework} is not supported, supported frameworks are {SUPPORTED_FRAMEWORK_TYPES}."
        )

    if framework == DDPCheckpointer._framework_name and DDP_ENABLED:
        return DDPCheckpointer.load(ckpt_path, checkpoint_state, fast_loading, strict, role)
    elif framework == FSDPCheckpointer._framework_name and FSDP_ENABLED:
        return FSDPCheckpointer.load(
            ckpt_path,
            checkpoint_state,
            fast_loading,
            strict,
            role,
            load_decomposed_model_optimizer=load_decomposed_model_optimizer,
        )
    elif framework == FSDP2Checkpointer._framework_name and FSDP2_ENABLED:
        return FSDP2Checkpointer.load(ckpt_path, checkpoint_state, fast_loading, strict, role)
    else:
        raise ValueError(f"Framework {framework} is not enabled.")
