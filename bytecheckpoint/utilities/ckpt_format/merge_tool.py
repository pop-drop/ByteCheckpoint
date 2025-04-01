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
from typing import Dict, List, Optional, Tuple

import torch
from torch.distributed.checkpoint.metadata import STORAGE_TYPES

from bytecheckpoint import BYTECHECKPOINT_GLOBAL_CONFIG
from bytecheckpoint.checkpointer.meta_type import FSDP_STR, STATE_DICT_TYPE, SUPPORTED_MERGING_FRAMEWORK_TYPES
from bytecheckpoint.io import bfile
from bytecheckpoint.io.bfile import list_files
from bytecheckpoint.utilities.ckpt_format.ckpt_loader import (
    CKPTLoader,
    load_tensor_shards_multiprocessing,
    load_tensor_shards_non_async,
)
from bytecheckpoint.utilities.ckpt_format.common_utils import get_ckpt_file_suffix
from bytecheckpoint.utilities.ckpt_format.fsdp_ckpt_utils import merge_fsdp_ckpt
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()


def bytecheckpoint_ckpt_to_pytorch_ckpt(
    save_path: str,
    output_path: str,
    framework: str = "fsdp",
    model_only: bool = False,
    optimizer_only: bool = False,
    return_dict: bool = False,
    fsdp_save_decomposed_model: bool = False,
    safetensors_format: bool = False,
    untie_embeddings: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> STATE_DICT_TYPE:
    """
    Merges and converts bytecheckpoint checkpoints to PyTorch torch.save/load compatible checkpoints.

    Args:
        save_path (str): Path to the directory containing the bytecheckpoint checkpoint files.
        output_path (str): Path where the converted PyTorch checkpoint(s) will be saved.
        framework (str, optional): The training framework used. Defaults to "fsdp".
        model_only (bool, optional): If True, only the model checkpoint will be converted. Defaults to False.
        optimizer_only (bool, optional): If True, only the optimizer checkpoint will be converted. Defaults to False.
        return_dict (bool, optional): If True, returns the merged checkpoint data as a dictionary. Defaults to False.
        fsdp_save_decomposed_model (bool, optional): Set true if the FSDP model is in flattened parameter format. Defaults to False.
        safetensors_format (bool, optional): If True, saves the checkpoint in safetensors format. Defaults to False.
        untie_embeddings (bool, optional): Set true when if embeddings in the saved checkpoint are untied. Defaults to False.
        dtype (Optional[torch.dtype]): The data type of the tensors. Defaults to None.

    Returns:
        dict: A dictionary containing the merged PyTorch checkpoint.
              Keys include "model" and/or "optimizer" depending on the input flags and available data.
    """

    assert framework in SUPPORTED_MERGING_FRAMEWORK_TYPES, f"Unsupported framework {framework} for merging checkpoint"

    bfile.makedirs(output_path)
    # Training configuration and tokenizer.
    merged_state_dicts = {}

    # Config args for DDP and FSDP2.
    if framework != FSDP_STR:
        fsdp_save_decomposed_model = False
        untie_embeddings = False
        logger.warning("Decomposed model representaiton is only supported for DDP and FSDP2")

    model_path = os.path.join(save_path, "model")
    if not optimizer_only:
        assert bfile.exists(model_path), f"{model_path} does not exist"
        file_suffix = get_ckpt_file_suffix(safetensors_format=safetensors_format)
        # Get output model path.
        output_model_path = os.path.join(output_path, "model" + file_suffix)
        merged_state_dicts["model"] = merge_distcp(
            bcp_checkpoint_dir=model_path,
            ckpt_save_path=output_model_path,
            return_dict=return_dict,
            fsdp_save_decomposed_model=fsdp_save_decomposed_model,
            safetensors_format=safetensors_format,
            untie_embeddings=untie_embeddings,
            dtype=dtype,
        )

    optimizer_path = os.path.join(save_path, "optimizer")
    if not model_only:
        assert bfile.exists(optimizer_path), f"{optimizer_path} does not exist"
        # Get optimizer path.
        output_optimizer_path = os.path.join(output_path, "optimizer.pt")
        logger.warning("Currently, we only support merge non-decomposed optimizer .distcp files into .pt format")
        merged_state_dicts["optimizer"] = merge_distcp(
            bcp_checkpoint_dir=optimizer_path,
            ckpt_save_path=output_optimizer_path,
            return_dict=return_dict,
            fsdp_save_decomposed_model=False,
            safetensors_format=False,
            untie_embeddings=False,
            dtype=dtype,
        )

    return merged_state_dicts


def merge_distcp(
    bcp_checkpoint_dir: str,
    ckpt_save_path: str,
    return_dict: bool = False,
    fsdp_save_decomposed_model: bool = False,
    safetensors_format: bool = False,
    untie_embeddings: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> Optional[STATE_DICT_TYPE]:
    """
    Merges distributed checkpoints into a single checkpoint.

    Args:
        bcp_checkpoint_dir (str): Directory containing the distributed checkpoints.
        ckpt_save_path (str): Path to save the merged checkpoint.
        return_dict (bool, optional): If True, returns the merged checkpoint as a dictionary. Defaults to False.
        fsdp_save_decomposed_model (bool, optional): If True, saves the FSDP model in a decomposed format. Defaults to False.
        safetensors_format (bool, optional): If True, saves the checkpoint in safetensors format. Defaults to False.
        untie_embeddings (bool, optional): If True, untie the embeddings in the saved checkpoint. Defaults to False.
        dtype (Optional[torch.dtype]): The data type of the tensors. Defaults to None.

    Returns:
        Optional[STATE_DICT_TYPE]: Merged checkpoint as a dictionary if return_dict is True, otherwise None.
    """
    loader = CKPTLoader(bcp_checkpoint_dir)

    meta = loader.load_metadata()
    files = list_files([bcp_checkpoint_dir])

    # Key: tensor fqn
    # Value: a list of tensor shards with (tensor, offset) tuple
    # Determine the loading mode.
    storage_data = meta.storage_data
    all_tensor_metadata = meta.all_tensor_metadata

    logger.info("start loading tensor shards")
    if BYTECHECKPOINT_GLOBAL_CONFIG.merge_num_io_worker == 1:
        # Disable multiprocessing.
        final_state_dict, tensor_shards_kv = load_tensor_shards_non_async(
            storage_data,
            all_tensor_metadata,
            loader,
            files,
            dtype,
        )

    else:
        # Multiprocessing load.
        final_state_dict, tensor_shards_kv = load_tensor_shards_multiprocessing(
            storage_data,
            all_tensor_metadata,
            loader,
            files,
            BYTECHECKPOINT_GLOBAL_CONFIG.merge_num_io_worker,
            dtype,
        )
    logger.info("finish loading tensor shards")

    logger.info("start merging tensor")
    for k in list(tensor_shards_kv.keys()):
        logger.info("merging tensor, key %s", k)
        shards_offsets = tensor_shards_kv[k]
        shards = []
        offsets = []
        # Don't merge ".step" in optimizer
        # it is an undimensional tensor
        for shard, offset in shards_offsets:
            shards.append(shard)
            offsets.append(offset)
        if "param_groups" in k or "opt_remain_group_idx" in k or offsets[0] == torch.Size([]):
            final_state_dict[k] = shards[0]
        else:
            merged_tensor = merge_tensors(k, shards, offsets, meta.state_dict_metadata)
            final_state_dict[k] = merged_tensor

        # Release memory by deleting shards_offsets and clearing lists
        del tensor_shards_kv[k]  # Delete to release memory
        del shards_offsets  # Delete to release memory
        shards.clear()  # Clear the shards list
        offsets.clear()  # Clear the offsets list

    final_state_dict = merge_fsdp_ckpt(
        meta=meta,
        final_state_dict=final_state_dict,
        fsdp_save_decomposed_model=fsdp_save_decomposed_model,
        untie_embeddings=untie_embeddings,
    )

    # Save merged checkpoint.
    if safetensors_format:
        from bytecheckpoint.utilities.ckpt_format.safetensors_utils import save_sharded_hf_checkpoint

        ckpt_save_path = os.path.dirname(ckpt_save_path)
        # Make memory contiguous before saving
        for key in list(final_state_dict.keys()):
            if not isinstance(final_state_dict[key], torch.Tensor):
                del final_state_dict[key]
            else:
                final_state_dict[key] = final_state_dict[key].contiguous()

        logger.info("Start saving ckpt to %s.", ckpt_save_path)
        os.makedirs(ckpt_save_path, exist_ok=True)
        save_sharded_hf_checkpoint(final_state_dict, ckpt_save_path)
        logger.info("Successfully finished saving merged ckpt to %s.", ckpt_save_path)
    else:
        logger.info(
            "Start saving merged checkpoint to local file %s.",
            ckpt_save_path,
        )
        # If the ckpt_save_path is a local path,
        # save the checkpoint to local path directly
        start_time = time.time()
        with open(ckpt_save_path, "wb") as file:
            torch.save(final_state_dict, file)

        logger.info(
            "Finished saving merged checkpoint to local file %s. cost time: %s s",
            ckpt_save_path,
            time.time() - start_time,
        )

    # Return merged checkpoint.
    if return_dict:
        return final_state_dict
    else:
        return {}


def merge_tensors(
    key: str,
    tensor_list: List[torch.Tensor],
    offsets: List[Tuple[int, ...]],
    metadata: Dict[str, STORAGE_TYPES] = None,
) -> torch.Tensor:
    if metadata is not None:
        num_dims = len(metadata[key].size)
    else:
        num_dims = len(tensor_list[0].size())  # Assuming all tensors have the same number of dimensions
        max_size = [0] * num_dims
        for tensor, offset in zip(tensor_list, offsets):
            for i in range(num_dims):
                max_size[i] = max(max_size[i], offset[i] + tensor.size(i))

    # [FP8 only] Process fp8 tensors as uint8 since "fill_cpu" not implemented for 'Float8_e4m3fn'
    for i in range(len(tensor_list)):
        if tensor_list[i].dtype == torch.float8_e4m3fn:
            tensor_list[i] = tensor_list[i].view(torch.uint8)

    # Create the base tensor initialized to zeros
    if metadata is not None:
        result_tensor = torch.zeros(metadata[key].size, dtype=tensor_list[0].dtype, device=tensor_list[0].device)
    else:
        result_tensor = torch.zeros(*max_size, dtype=tensor_list[0].dtype, device=tensor_list[0].device)

    # Place each tensor at its offset
    for tensor, offset in zip(tensor_list, offsets):
        slices = tuple(slice(offset[dim], offset[dim] + tensor.size(dim)) for dim in range(num_dims))
        result_tensor[slices] = tensor

    return result_tensor
