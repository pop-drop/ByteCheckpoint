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

import re
from typing import Dict, List, Tuple

import torch
from torch.distributed.checkpoint._nested_dict import unflatten_state_dict

from bytecheckpoint.checkpointer.meta_type import STATE_DICT_TYPE, Metadata
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()


def find_overlap(str1: str, str2: str):
    """
    Find the maximum overlap between the end of str1 and the start of str2.

    Args:
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.

    Returns:
        str: The maximum overlap string.
    """
    max_overlap = ""

    for i in range(1, len(str1) + 1):
        suffix = str1[-i:]
        if re.match(f"^{re.escape(suffix)}", str2):
            max_overlap = suffix

    return max_overlap


def fsdp_split_and_unflat_model_param(
    unflatten_state_dict: STATE_DICT_TYPE,
    state_dict: STATE_DICT_TYPE,
    flat_param_to_fqns: Dict[str, List[str]],
    is_padding_mask: Dict[str, List[bool]],
    numels_with_padding: Dict[str, Tuple[int, ...]],
    shapes: Dict[str, Tuple[torch.Size, ...]],
) -> None:
    """
    Split and unflatten the model parameters in the state dictionary according to the provided metadata.

    Args:
        unflatten_state_dict (STATE_DICT_TYPE): The dictionary to store the unflattened state.
        state_dict (STATE_DICT_TYPE): The state dictionary containing the flattened parameters.
        flat_param_to_fqns (Dict[str, List[str]]): A mapping from flat parameter keys to fully qualified names.
        is_padding_mask (Dict[str, List[bool]]): A mask indicating whether each split is padding.
        numels_with_padding (Dict[str, Tuple[int, ...]]): The number of elements in each split, including padding.
        shapes (Dict[str, Tuple[torch.Size, ...]]): The shapes of each split after removing padding.

    Returns:
        None: The function modifies the unflatten_state_dict in-place.
    """
    from torch.distributed.fsdp._unshard_param_utils import FLAT_PARAM

    keys_to_delete = []

    for key, tensor in state_dict.items():
        if key not in flat_param_to_fqns:
            logger.info("There is no fsdp flat param meta data related to %s", key)
            unflatten_state_dict[key] = tensor
            continue
        keys_to_delete.append(key)
        no_flat_param_key = key.replace(FLAT_PARAM, "")
        flat_param_to_fqns_per_key = flat_param_to_fqns[key]
        is_padding_mask_per_key = is_padding_mask[key]
        numels_with_padding_per_key = numels_with_padding[key]
        shapes_per_key = shapes[key]

        splits = torch.split(tensor, numels_with_padding_per_key, dim=0)
        idx = 0
        for split, is_padding in zip(splits, is_padding_mask_per_key):
            if is_padding:
                continue
            # find_overlap is useless now, but we keep it now in order to be compatible with old version ckpt
            len_overlap = len(find_overlap(no_flat_param_key, flat_param_to_fqns_per_key[idx]))
            unflatten_state_dict[f"{no_flat_param_key}{flat_param_to_fqns_per_key[idx][len_overlap:]}"] = split.view(
                shapes_per_key[idx]
            )
            idx += 1


def merge_fsdp_ckpt(
    meta: Metadata,
    final_state_dict: Dict[str, torch.Tensor],
    fsdp_save_decomposed_model: bool,
    untie_embeddings: bool = False,
) -> STATE_DICT_TYPE:
    """
    Merge the FSDP (Fully Sharded Data Parallel) checkpoint state dictionary.

    Args:
        meta (Metadata): Metadata containing planner data and user-defined dictionary.
        final_state_dict (Dict[str, torch.Tensor]): The final state dictionary to be merged.
        fsdp_save_decomposed_model (bool): Flag indicating whether the model is saved in a decomposed representation.
        untie_embeddings (bool, optional): Flag indicating whether to untie the embeddings. Defaults to False.

    Returns:
        STATE_DICT_TYPE: The merged state dictionary.
    """
    # Unflatten state dict.
    planner_data = meta.planner_data
    final_state_dict = unflatten_state_dict(final_state_dict, planner_data)

    # Merge model in decomposed representation.
    if fsdp_save_decomposed_model:
        from bytecheckpoint.checkpointer.meta_type import (
            FSDP_FLAT_PARAM_TO_FQNS,
            FSDP_IS_PADDING_MASK,
            FSDP_NUMELS_WITH_PADDING,
            FSDP_SHAPES,
            MODEL_DICT_KEYS,
        )

        user_defined_dict = meta.user_defined_dict
        logger.debug("final_state_dict keys: %s", final_state_dict.keys())
        logger.debug("meta_user_defined_dict: %s", user_defined_dict)
        model_dict_keys = user_defined_dict.pop(MODEL_DICT_KEYS, None)
        unflatten_final_state_dict = {}
        if model_dict_keys is not None:
            for model_key in model_dict_keys:
                flat_param_to_fqns = user_defined_dict[model_key][FSDP_FLAT_PARAM_TO_FQNS]
                is_padding_mask = user_defined_dict[model_key][FSDP_IS_PADDING_MASK]
                numels_with_padding = user_defined_dict[model_key][FSDP_NUMELS_WITH_PADDING]
                shapes = user_defined_dict[model_key][FSDP_SHAPES]
                unflatten_final_state_dict[model_key] = {}
                fsdp_split_and_unflat_model_param(
                    unflatten_final_state_dict[model_key],
                    final_state_dict[model_key],
                    flat_param_to_fqns,
                    is_padding_mask,
                    numels_with_padding,
                    shapes,
                )
        else:
            flat_param_to_fqns = user_defined_dict[FSDP_FLAT_PARAM_TO_FQNS]
            is_padding_mask = user_defined_dict[FSDP_IS_PADDING_MASK]
            numels_with_padding = user_defined_dict[FSDP_NUMELS_WITH_PADDING]
            shapes = user_defined_dict[FSDP_SHAPES]
            fsdp_split_and_unflat_model_param(
                unflatten_final_state_dict,
                final_state_dict,
                flat_param_to_fqns,
                is_padding_mask,
                numels_with_padding,
                shapes,
            )

        final_state_dict = unflatten_final_state_dict
        logger.debug("unflatten final_state_dict keys, %s", final_state_dict.keys())

    # Handle tied weights.
    for k in list(final_state_dict.keys()):
        v = final_state_dict[k]
        # embeddings
        if k == "transformer.wte.weight":
            if not untie_embeddings:
                final_state_dict["lm_head.weight"] = v.clone()
    return final_state_dict
