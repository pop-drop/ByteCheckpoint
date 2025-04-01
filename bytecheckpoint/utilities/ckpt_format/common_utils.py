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

import inspect
import os
from typing import Optional

from bytecheckpoint.checkpointer.meta_type import (
    _DIRECTORY_FORMAT,
    BYTECHECKPOINT_LOCAL_TEMP_DIR_FOR_UPLOADING_PREFIX,
    SHM_PATH,
)
from bytecheckpoint.io import bfile
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()


def get_ckpt_file_suffix(safetensors_format: bool = False) -> str:
    """
    Return the appropriate file suffix for a checkpoint file based on the format.

    Args:
        safetensors_format (bool, optional): A boolean indicating whether to use the safetensors format.
                                             Defaults to False.

    Returns:
        str: The file suffix for the checkpoint file. If safetensors_format is True, returns '.safetensors'.
             Otherwise, returns '.pt'.
    """
    if safetensors_format:
        return ".safetensors"
    else:
        return ".pt"


def extract_layer_id(key: str) -> int:
    """
    Extract the layer ID from a given key string.

    This function uses a regular expression to search for a layer ID within the provided key.
    The layer ID is expected to be in the format of 'language_model.encoder.layers.<id>.' within the key.

    Args:
        key (str): The key string from which to extract the layer ID.

    Returns:
        int: The extracted layer ID as an integer if found. If no layer ID is found, returns None.
    """
    import re

    # Define a regex pattern to capture the layer_id
    pattern = r"(?:^|[.])language_model\.encoder\.layers\.(\d+)\."

    # Search for the pattern in the key
    match = re.search(pattern, key)

    # Return the extracted layer_id as an integer if found, otherwise None
    return int(match.group(1)) if match else None


def bytecheckpoint_cleanup_shm_ckpt_dir() -> None:
    """
    Clean up the shared memory (SHM) directory used for uploading checkpoints when the training process crashes.
    This function searches for all directories in the SHM path that match the given prefix and deletes them.

    Args:
        None

    Returns:
        None
    """
    import glob
    import shutil

    dirs = glob.glob(f"{SHM_PATH}/{BYTECHECKPOINT_LOCAL_TEMP_DIR_FOR_UPLOADING_PREFIX}*")
    # Delete all dir
    for dir in dirs:
        try:
            shutil.rmtree(dir)
            logger.info("Deleted: %s", dir)
        except Exception as e:
            logger.warning("Error deleting %s: %s", dir, e)


def find_latest_ckpt_path(path: str) -> Optional[str]:
    """
    Get the latest complete checkpoint path under the given checkpoint root path.

    This function retrieves the latest checkpoint iteration number from the 'latest_checkpointed_iteration.txt' file.
    It then constructs the path to the latest complete checkpoint using this iteration number.

    Args:
        path (str): The root path of the checkpoints.

    Returns:
        str: The path to the latest complete checkpoint, or None if no valid checkpoint is found.
    """
    # Get latest complete checkpoint path under checkpoint root path
    # This function gets latest_checkpointed_iteration from latest_checkpointed_iteration.txt
    # Then return the latest complete checkpoint path
    # Example:
    # path = "/checkpoint_dir/"
    # This path contains:
    # /checkpoint_dir/global_step_100
    # /checkpoint_dir/global_step_200
    # /checkpoint_dir/global_step_300 (failed checkpoint)
    # /checkpoint_dir/latest_checkpointed_iteration.txt
    # This function returns:
    # /checkpoint_dir/global_step_200

    if path is None:
        return None

    tracker_file = get_checkpoint_tracker_filename(path)
    if not bfile.exists(tracker_file):
        logger.warning("Checkpoint does not exist: %s", tracker_file)
        return None

    with bfile.BFile(tracker_file, "rb", skip_encryption=True) as f:
        iteration = int(f.read().decode())
    ckpt_path = os.path.join(path, _DIRECTORY_FORMAT.format(iteration))
    if not bfile.exists(ckpt_path):
        logger.warning("Checkpoint does not exist: %s", ckpt_path)
        return None

    logger.info("Found checkpoint: %s", ckpt_path)
    return ckpt_path


def get_checkpoint_tracker_filename(root_path: str) -> str:
    """
    Generate the full path to the checkpoint tracker file.

    The checkpoint tracker file is used to record the latest checkpoint iteration number during training.
    This allows the training process to resume from the last saved checkpoint.

    Args:
        root_path (str): The root directory path where the checkpoint tracker file is located.

    Returns:
        str: The full path to the checkpoint tracker file, which is 'latest_checkpointed_iteration.txt'
             in the specified root directory.
    """
    return os.path.join(root_path, "latest_checkpointed_iteration.txt")


def check_ckpt_is_bytecheckpoint(path: str, check_by_sub_folders: bool = None) -> bool:
    """
    Check if the given path is a valid ByteCheckpoint checkpoint directory.

    A ByteCheckpoint checkpoint directory should contain at least a 'model' subfolder.
    Optionally, it may also contain 'optimizer', 'extra_state', and 'loader' subfolders.

    Args:
        path (str): The path to the directory to check.
        check_by_sub_folders (list, optional): A list of subfolders to check for existence.
                                               Defaults to ['model'].

    Returns:
        bool: True if all specified subfolders exist in the given path, False otherwise.
    """
    # bytecheckpoint ckpt folder should contain:
    # ${path}/model
    # [optional] ${path}/optimizer
    # [optional] ${path}/extra_state
    # [optional] ${path}/loader
    # check by model sub folder by default, however, users are still allowed to customize, such as using optimizer or
    # extra_state sub folder to check
    if check_by_sub_folders is None:
        check_by_sub_folders = ["model"]
    if path is None:
        return False
    for sub_folder in check_by_sub_folders:
        target_file_path = os.path.join(path, sub_folder)
        if not bfile.exists(target_file_path):
            return False
    return True


def model_optimizer_completeness(ckpt_path: str, num_of_processes: int) -> bool:
    """
    Check the completeness of the model and optimizer components in a checkpoint directory.

    This function verifies if the specified checkpoint directory contains both the 'model' and 'optimizer'
    components, along with their respective '.metadata' files and distributed checkpoint files for each process.

    Args:
        ckpt_path (str): The path to the checkpoint directory.
        num_of_processes (int): The number of processes for which distributed checkpoint files should exist.

    Returns:
        bool: True if all components and files are present, False otherwise.
    """
    for component in ["model", "optimizer"]:
        subpath = os.path.join(ckpt_path, component)
        if not os.path.exists(subpath):
            print(f"{component} subpath does not exist")
            return False
        if not os.path.exists(os.path.join(subpath, ".metadata")):
            print(f"{component} subpath does not contain .metadata")
            return False
        for i in range(num_of_processes):
            if not os.path.exists(os.path.join(subpath, f"__{i}_0.distcp")):
                print(f"{component} subpath does not contain __{i}_0.distcp")
                return False
    return True


def callback_has_params(callback_fn):
    """
    Check if the given callback function has any parameters.

    Args:
        callback_fn (callable): The callback function to check.

    Returns:
        bool: True if the callback function has parameters, False otherwise.
    """
    signature = inspect.signature(callback_fn)
    return len(signature.parameters) > 0
