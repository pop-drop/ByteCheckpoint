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

from typing import Optional

from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()

from bytecheckpoint.checkpointer.ddp_checkpointer import DDPCheckpointer
from bytecheckpoint.checkpointer.fsdp_checkpointer import FSDPCheckpointer

try:
    from bytecheckpoint.checkpointer.fsdp2_checkpointer import FSDP2Checkpointer
except Exception as e:
    logger.warning("Failed to import FSDP2Checkpointer")
    FSDP2_ENABLED = False
    pass

from bytecheckpoint.checkpointer.meta_type import SUPPORTED_FRAMEWORK_TYPES, CheckpointState


def save(
    ckpt_path: str,
    checkpoint_state: CheckpointState,
    framework: str,
    fast_saving=False,
    global_steps: Optional[int] = None,
    callback: Optional[callable] = None,
    role: str = None,
    ignore_append_global_steps_to_folder: Optional[bool] = False,
    *,
    save_decomposed_model_optimizer: bool = False,
):
    """
    Save API.

    Args:
        ckpt_path (str): The path where the checkpoint will be saved.
        checkpoint_state (CheckpointState): The state of the checkpoint to be saved.
        framework (str): The framework used for saving the checkpoint. Must be one of the supported framework types.
        fast_saving (bool, optional): Whether to save the checkpoint asynchronously. Defaults to False.
        global_steps (Optional[int], optional): The global steps of the checkpoint. Defaults to None.
        callback (Optional[callable], optional): A callback function to be called after saving the checkpoint. Defaults to None.
        role (str, optional): The role of the model and optimizer to be saved, e.g. 'actor' or 'critic' in RL. Defaults to None.
        ignore_append_global_steps_to_folder (Optional[bool], optional): Whether to ignore appending global steps to the folder name. Defaults to False.
        save_decomposed_model_optimizer (bool, optional): Whether to save the decomposed model optimizer. Defaults to False.

        NOTE: `save_decomposed_model_optimizer` is only valid for FSDP checkpoints.

    Returns:
        The result of the underlying checkpointer's save method.
    """
    if framework not in SUPPORTED_FRAMEWORK_TYPES:
        raise ValueError(
            f"Framework {framework} is not supported, supported frameworks are {SUPPORTED_FRAMEWORK_TYPES}."
        )

    if framework == DDPCheckpointer._framework_name and DDP_ENABLED:
        return DDPCheckpointer.save(
            ckpt_path,
            checkpoint_state,
            fast_saving,
            global_steps,
            callback,
            role,
            ignore_append_global_steps_to_folder,
        )
    elif framework == FSDPCheckpointer._framework_name and FSDP_ENABLED:
        return FSDPCheckpointer.save(
            ckpt_path,
            checkpoint_state,
            fast_saving,
            global_steps,
            callback,
            role,
            ignore_append_global_steps_to_folder,
            save_decomposed_model_optimizer=save_decomposed_model_optimizer,
        )
    elif framework == FSDP2Checkpointer._framework_name and FSDP2_ENABLED:
        return FSDP2Checkpointer.save(
            ckpt_path,
            checkpoint_state,
            fast_saving,
            global_steps,
            callback,
            role,
            ignore_append_global_steps_to_folder,
        )
    else:
        raise ValueError(f"Framework {framework} is not enabled.")
