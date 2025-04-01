################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
################################################################################

from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, no_type_check

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import distributed_c10d
from torch.distributed._composable_state import _get_module_state
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
)
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    StateDictSettings,  # noqa: F401
    StateDictType,
)
from torch.distributed.fsdp._common_utils import (
    FSDP_PREFIX,
    _apply_to_modules,
    _FSDPState,
    _get_module_fsdp_state_if_fully_sharded_module,
    _get_param_to_fqns,
    _has_fsdp_params,
    _module_handle,
    _named_parameters_with_duplicates,
    clean_tensor_name,
)
from torch.distributed.fsdp._optim_utils import _is_named_optimizer
from torch.distributed.fsdp._runtime_utils import (
    _lazy_init,
)
from torch.distributed.fsdp._unshard_param_utils import FLAT_PARAM
from torch.distributed.utils import _replace_by_prefix

from ...checkpointer.meta_type import (
    FSDP_FLAT_PARAM_META,
    FSDP_FLAT_PARAM_TO_FQNS,
    FSDP_IS_PADDING_MASK,
    FSDP_NUMELS_WITH_PADDING,
    FSDP_OPTIMIZER_REQUIRES_GRAD_NUMEL,
    FSDP_SHAPES,
)
from ...utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()


def find_tied_weights(model: FSDP):
    """
    Find the tied weights in a Fully Sharded Data Parallel (FSDP) model.

    Tied weights are parameters that are shared across multiple parts of the model.
    This function identifies these tied weights by comparing the set of all parameter names
    (including duplicates) with the set of unique parameter names.

    Args:
        model (FSDP): The FSDP model in which to find tied weights.

    Returns:
        Optional[List[str]]: A list of tied weight names, with the FSDP prefix removed.
            If no tied weights are found, returns None.
    """
    all_params = model.named_parameters(remove_duplicate=False)
    all_param_names = {name for name, _ in all_params}
    unique_params = model.named_parameters()
    unique_param_names = {name for name, _ in unique_params}
    tied_weights = list(all_param_names - unique_param_names)
    tied_weights = [_remove_sub_string(name, FSDP_PREFIX) for name in tied_weights]
    print(
        f"ByteCheckpoint FSDP hacker found tied weights: {tied_weights}, which will be ignored when using ByteCheckpoint FSDP flatten ckpt mode"
    )
    if len(tied_weights) == 0:
        return None
    return tied_weights


def init_fsdp_hack(tied_weights=None):
    """
    Initialize the hacks for Fully Sharded Data Parallel (FSDP) in PyTorch.

    This function replaces several methods and hooks in the FSDP implementation
    with custom implementations to support specific features such as zero-communication
    optimizer state dict retrieval and compatibility with device mesh.

    Args:
        tied_weights (Optional[List[str]]): A list of tied weights to be removed from the state_dict.
            Defaults to None.
    """
    # TODO: Get optimizer state dict with zero communication.
    FSDP.optim_state_dict = staticmethod(hack_optim_state_dict)
    FSDP.optim_state_dict_to_load = staticmethod(hack_optim_state_dict_to_load)
    torch.distributed.fsdp._state_dict_utils._local_post_state_dict_hook = partial(
        hack_local_post_state_dict_hook,
        tied_weights=tied_weights,
    )
    torch.distributed.fsdp._state_dict_utils._local_pre_load_state_dict_hook = hack_local_pre_load_state_dict_hook
    # To be compatible with device mesh.
    torch.distributed.fsdp._state_dict_utils._set_use_dtensor = hack_set_use_dtensor
    torch.distributed.fsdp._optim_utils._set_optim_use_dtensor = hack_set_optim_use_dtensor
    torch.distributed.fsdp.fully_sharded_data_parallel._set_optim_use_dtensor = hack_set_optim_use_dtensor


def hack_optim_state_dict(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    optim_state_dict: Optional[Dict[str, Any]] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    Generate an optimizer state dictionary for FSDP (Fully Sharded Data Parallel) modules.

    This function creates a state dictionary for the optimizer, taking into account
    the parameters and states of FSDP modules within the provided model. It processes the
    optimizer state based on the state dictionary type settings of the FSDP module.

    Args:
        model (torch.nn.Module): The model containing FSDP modules.
        optim (torch.optim.Optimizer): The optimizer used for training.
        optim_state_dict (Optional[Dict[str, Any]]): The optimizer state dictionary. Defaults to None.
        group (Optional[dist.ProcessGroup]): The process group for distributed training. Defaults to None.

    Returns:
        Dict[str, Any]: A state dictionary for the optimizer.
    """
    state_dict_settings = FSDP.get_state_dict_type(model)
    if optim_state_dict is None:
        optim_state_dict = optim.state_dict()
    if state_dict_settings.state_dict_type == StateDictType.LOCAL_STATE_DICT:
        return _local_optim_state_dict_impl(
            model=model,
            optim=optim,
            optim_state_dict=optim_state_dict,
            group=group,
        )
    return FSDP._optim_state_dict_impl(
        model=model,
        optim=optim,
        optim_state_dict=optim_state_dict,
        optim_input=None,
        rank0_only=getattr(state_dict_settings.optim_state_dict_config, "rank0_only", False),
        full_state_dict=state_dict_settings.state_dict_type == StateDictType.FULL_STATE_DICT,
        group=group,
    )


def hack_optim_state_dict_to_load(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    optim_state_dict: Dict[str, Any],
    is_named_optimizer: bool = False,
    load_directly: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    Generate an optimizer state dictionary to load for FSDP (Fully Sharded Data Parallel) modules.

    This function creates a state dictionary for the optimizer to load, taking into account
    the state dictionary type settings of the FSDP module. It processes the optimizer state
    based on whether the state dictionary type is local or full.

    Args:
        model (torch.nn.Module): The model containing FSDP modules.
        optim (torch.optim.Optimizer): The optimizer used for training.
        optim_state_dict (Dict[str, Any]): The optimizer state dictionary to load.
        is_named_optimizer (bool, optional): Whether the optimizer is a named optimizer. Defaults to False.
        load_directly (bool, optional): Whether to load the state dictionary directly into the optimizer. Defaults to False.
        group (Optional[dist.ProcessGroup], optional): The process group for distributed training. Defaults to None.

    Returns:
        Dict[str, Any]: A state dictionary for the optimizer to load.
    """
    state_dict_settings = FSDP.get_state_dict_type(model)
    if state_dict_settings.state_dict_type == StateDictType.LOCAL_STATE_DICT:
        assert not is_named_optimizer, "Not suppport named optimizer now"
        result = _local_optim_state_dict_to_load_impl(
            model=model,
            optim=optim,
            optim_sharded_state_dict=optim_state_dict,
            group=group,
        )
        return result
    result = FSDP._optim_state_dict_to_load_impl(
        optim_state_dict=optim_state_dict,
        model=model,
        optim_input=None,
        optim=optim,
        full_state_dict=(state_dict_settings.state_dict_type == StateDictType.FULL_STATE_DICT),
        rank0_only=getattr(state_dict_settings.optim_state_dict_config, "rank0_only", False),
        is_named_optimizer=is_named_optimizer,
        group=group,
    )
    if load_directly:
        optim.load_state_dict(result)
    return result


def _local_optim_state_dict_impl(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    optim_state_dict: Optional[Dict[str, Any]] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    Generate a local optimizer state dictionary for FSDP (Fully Sharded Data Parallel) modules.

    This function creates a sharded state dictionary for the optimizer, taking into account
    the parameters and states of FSDP modules within the provided model. It processes the
    optimizer state, handles parameter grouping, and constructs ShardedTensors for state values
    where applicable.

    Args:
        model (torch.nn.Module): The model containing FSDP modules.
        optim (torch.optim.Optimizer): The optimizer used for training.
        optim_state_dict (Optional[Dict[str, Any]]): The optimizer state dictionary. Defaults to None.
        group (Optional[dist.ProcessGroup]): The process group for distributed training. Defaults to None.

    Returns:
        Dict[str, Any]: A sharded state dictionary for the optimizer.
    """
    logger.info("into bytecheckpoint hack_optim_state_dict")
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(model)
    _lazy_init(fsdp_state, model)
    assert fsdp_state._is_root, "User should call optim_state_dict with root FSDP module"
    # Need to reset flat param grad?
    # _reset_flat_param_grad_info_if_needed(traversal_utils._get_fsdp_handles(model))
    use_orig_params = fsdp_state._use_orig_params
    assert use_orig_params, "Now only support FSDP when use_orig_params==True"

    all_FSDP_modules = FSDP.fsdp_modules(model)
    assert all(use_orig_params == m._use_orig_params for m in all_FSDP_modules), (
        "Not all FSDP modules have the same _use_orig_params value"
    )

    is_named_optimizer = _is_named_optimizer(optim_state_dict)
    assert not is_named_optimizer, "Not suppport named optimizer now"

    optim_sharded_state_dict: Dict[str, Any] = {"state": {}}
    # param_to_fqns is Dict[tensor, List[str]].
    param_to_fqns = _get_param_to_fqns(model)
    fsdp_state_to_fqn_prefix = _get_fsdp_state_to_fqn_prefix(model)
    param_to_param_key: Dict[nn.Parameter, int] = {}
    pid = 0
    for param_group in optim.param_groups:
        for param in param_group["params"]:
            assert param not in param_to_param_key, "Not supported that one parameter register in multiple param groups"
            param_to_param_key[param] = pid
            pid += 1
    # Get total param numels for verifcation.
    # This logic must be here.
    for module in all_FSDP_modules:
        fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
        if not _has_fsdp_params(fsdp_state, module):
            continue
        prefix = fsdp_state_to_fqn_prefix[fsdp_state]
        flat_param = _module_handle(fsdp_state, module).flat_param
        total_fsdp_requires_grad_param_numel = 0
        for param_idx, numel in enumerate(flat_param._numels):
            if flat_param._params[param_idx].requires_grad and flat_param._params[param_idx] in param_to_param_key:
                total_fsdp_requires_grad_param_numel += numel
        optim_sharded_state_dict["state"][f"{prefix}{FLAT_PARAM}.{FSDP_OPTIMIZER_REQUIRES_GRAD_NUMEL}"] = torch.tensor(
            total_fsdp_requires_grad_param_numel
        )
    # Deepcopy param_groups.
    if "param_groups" in optim_state_dict:
        optim_sharded_state_dict["param_groups"] = deepcopy(optim_state_dict["param_groups"])
    # If optim_state_dict["state"] is empty， return early.
    if optim_state_dict["state"] == {}:
        logger.warning("Optimizer state is empty! ByteCheckpoint will save empty optimizer state in ckpt")
        return optim_sharded_state_dict
    # Get all state names.
    random_param = next(iter(optim_state_dict["state"]))
    all_state_name = optim_state_dict["state"][random_param].keys()

    for module in all_FSDP_modules:
        fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
        if not _has_fsdp_params(fsdp_state, module):
            continue
        assert _module_handle(fsdp_state, module), "Should have returned early"
        flat_param = _module_handle(fsdp_state, module).flat_param
        prefix = fsdp_state_to_fqn_prefix[fsdp_state]
        for state_name in all_state_name:
            local_shards: List[Shard] = []
            for shard_param_info, param, param_info in zip(
                flat_param._shard_param_infos, flat_param._params, flat_param._param_infos
            ):
                if not param.requires_grad:
                    continue
                if param not in param_to_param_key:
                    # Skip params that not registered in the optimizer
                    continue
                if shard_param_info.in_shard:
                    state_value = optim_state_dict["state"][param_to_param_key[param]][state_name]
                    if not torch.is_tensor(state_value) or state_value.dim() == 0:
                        optim_sharded_state_dict["state"][f"{param_to_fqns[param][0]}.{state_name}"] = state_value
                        continue
                    # No need to save padding. ByteCheckpoint will not read padding when loading optimizer.
                    assert len(state_value) == len(param)
                    # flat_param._sharded_size == flat_param.numel()？ YES
                    shard_offset = flat_param.numel() * fsdp_state.rank + shard_param_info.offset_in_shard
                    local_shards.append(Shard.from_tensor_and_offsets(state_value, [shard_offset], fsdp_state.rank))
            if len(local_shards) == 0:
                # state.step will reach here
                continue
            full_numel = flat_param._unpadded_unsharded_size.numel()  # type: ignore[attr-defined]
            global_ranks_in_pg = dist.get_process_group_ranks(fsdp_state.process_group)
            device_type = distributed_c10d._get_pg_default_device(fsdp_state.process_group).type
            placements = [
                f"rank:{global_rank}/{device_type}:{global_rank % fsdp_state._device_handle.device_count()}"
                for global_rank in global_ranks_in_pg
            ]
            sharding_spec = ChunkShardingSpec(
                dim=0,
                placements=placements,
            )
            sharded_tensor = ShardedTensor.__new__(
                ShardedTensor,
                sharding_spec,
                full_numel,
                dtype=flat_param.dtype,
                layout=flat_param.layout,
                pin_memory=flat_param.is_pinned(),
                requires_grad=flat_param.requires_grad,
            )
            sharded_tensor._prepare_init(process_group=fsdp_state.process_group, init_rrefs=False)
            # attach local_shards to the ShardedTensor created
            sharded_tensor._local_shards = local_shards
            # run post initialization, i.e. map registration, rpc initialization
            sharded_tensor._post_init()

            optim_sharded_state_dict["state"][f"{prefix}{FLAT_PARAM}.{state_name}"] = sharded_tensor
    return optim_sharded_state_dict


def _get_fsdp_state_to_fqn_prefix(model: torch.nn.Module) -> Dict[nn.Parameter, List[str]]:
    """
    Generates a mapping from FSDP states to their corresponding fully qualified name (FQN) prefixes.

    Args:
        model (torch.nn.Module): The input model for which the mapping is to be generated.

    Returns:
        Dict[nn.Parameter, List[str]]: A dictionary mapping each FSDP state to a list of FQN prefixes.
    """

    def module_fn(module, prefix, tree_level, fsdp_state_to_fqn_prefix):
        fqn_prefix = clean_tensor_name(prefix)
        state = _get_module_state(module)
        if isinstance(state, _FSDPState):
            fsdp_state_to_fqn_prefix[state] = fqn_prefix

    def return_fn(fsdp_state_to_fqn_prefix):
        return fsdp_state_to_fqn_prefix

    fsdp_state_to_fqn_prefix: Dict[_FSDPState, List[str]] = {}
    return _apply_to_modules(
        model,
        module_fn,
        return_fn,
        [key for key, _ in _named_parameters_with_duplicates(model)],
        fsdp_state_to_fqn_prefix,
    )


def _local_optim_state_dict_to_load_impl(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    optim_sharded_state_dict: Optional[Dict[str, Any]] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    Generate a local optimizer state dictionary to load for FSDP (Fully Sharded Data Parallel) modules.

    This function creates a state dictionary for the optimizer to load, taking into account
    the sharded state dictionary of the optimizer and the current state of the optimizer.
    It checks if the parameter group sizes in the loaded state dictionary match those of the optimizer,
    and updates the optimizer's parameter groups with the values from the loaded state dictionary.

    Args:
        model (torch.nn.Module): The model containing FSDP modules.
        optim (torch.optim.Optimizer): The optimizer used for training.
        optim_sharded_state_dict (Optional[Dict[str, Any]]): The sharded optimizer state dictionary to load. Defaults to None.
        group (Optional[dist.ProcessGroup]): The process group for distributed training. Defaults to None.

    Returns:
        Dict[str, Any]: A state dictionary for the optimizer to load.
    """
    logger.info("into bytecheckpoint hack_optim_state_dict_to_load")
    ckpt_param_lens = (len(g["params"]) for g in optim_sharded_state_dict["param_groups"])
    param_lens = (len(g["params"]) for g in optim.state_dict()["param_groups"])
    if any(p_len != s_len for p_len, s_len in zip(param_lens, ckpt_param_lens)):
        raise ValueError(
            "loaded state dict contains a parameter group that doesn't match the size of optimizer's group"
        )
    # deepcopy param_groups
    for idx, group in enumerate(optim_sharded_state_dict["param_groups"]):
        for key, value in group.items():
            if key != "params":
                optim.param_groups[idx][key] = value


def _remove_sub_string(str, substr):
    return str.replace(substr, "")


@no_type_check
def hack_local_post_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: Dict[str, Any],
    prefix: str,
    tied_weights: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    This hook creates a ShardedTensor from the local flat_param without copying the data.
    The underlying storage of the ShardedTensor is the same as the flat_param.

    Args:
        module (nn.Module): The module for which the state_dict is being created.
        fsdp_state (_FSDPState): The state of the FSDP module.
        state_dict (Dict[str, Any]): The state_dict of the module.
        prefix (str): The prefix to be removed from the state_dict keys.
        tied_weights (Optional[List[str]]): A list of tied weights to be removed from the state_dict.

    Returns:
        Dict[str, Any]: The modified state_dict with the flat_param replaced by a ShardedTensor.
    """
    assert fsdp_state._use_orig_params, "Now only support FSDP when use_orig_params==True"
    if fsdp_state._is_root:
        logger.info("into bytecheckpoint hack_local_post_state_dict_hook")
    # remove ._fsdp_wrapped_module
    # for example: _fsdp_wrapped_module.layers.0._fsdp_wrapped_module._flat_param to _fsdp_wrapped_module.layers.0._flat_param
    _replace_by_prefix(state_dict, f"{prefix}{FSDP_PREFIX}", prefix)
    if not _has_fsdp_params(fsdp_state, module):
        return state_dict

    # state_dict[f"{prefix}{FLAT_PARAM}"] exists and has the same tensor
    # value as the flat_param but it is a pure Tensor because
    # nn.Module.state_dict() will detach the parameter. Therefore, we need
    # to get flat_param to get the metadata.
    assert _module_handle(fsdp_state, module), "Should have returned early"
    flat_param = _module_handle(fsdp_state, module).flat_param
    # fqns：net1.weight...
    fqns = flat_param._fqns
    is_padding_mask = flat_param._is_padding_mask
    numels_with_padding = flat_param._numels_with_padding
    shapes = flat_param._shapes
    # Constructs a ShardedTensor from the flat_param "without" padding.
    # Removing the padding allows users to change the number of ranks
    # when loading the local_state_dict.
    full_numel = flat_param._unpadded_unsharded_size.numel()  # type: ignore[attr-defined]
    shard_offset = flat_param.numel() * fsdp_state.rank
    valid_data_size = flat_param.numel() - flat_param._shard_numel_padded
    if valid_data_size > 0:
        # If FlatParameter is returned, FlatParameter._local_shard cause a
        # pickling issue (can be torch.save but not torch.load). Since there
        # is no benefit for state_dict to return the actual FlatParameter class,
        # a view (which is a tensor) of the FlatParameter will be returned.
        flat_param = flat_param[:valid_data_size].view(valid_data_size)
        local_shards = [Shard.from_tensor_and_offsets(flat_param, [shard_offset], fsdp_state.rank)]
    else:
        local_shards = []
    # Create a ShardedTensor without invoking communication.
    # init_from_local_shards will invoke all_gather_object
    # sharded_tensor = init_from_local_shards(local_shards, full_numel, process_group=fsdp_state.process_group)  # type: ignore[assignment]
    # We remove all_gather_object and do verification in bytecheckpoint global plan
    global_ranks_in_pg = dist.get_process_group_ranks(fsdp_state.process_group)
    device_type = distributed_c10d._get_pg_default_device(fsdp_state.process_group).type
    # Actually we do not care placements' content.
    # TODO: Design a simple ShardedTensor for ByteCheckpoint FSDPCheckpointer
    placements = [
        f"rank:{global_rank}/{device_type}:{global_rank % fsdp_state._device_handle.device_count()}"
        for global_rank in global_ranks_in_pg
    ]
    sharding_spec = ChunkShardingSpec(
        dim=0,
        placements=placements,
    )
    sharded_tensor = ShardedTensor.__new__(
        ShardedTensor,
        sharding_spec,
        full_numel,
        dtype=flat_param.dtype,
        layout=flat_param.layout,
        pin_memory=flat_param.is_pinned(),
        requires_grad=flat_param.requires_grad,
    )
    sharded_tensor._prepare_init(process_group=fsdp_state.process_group, init_rrefs=False)
    # attach local_shards to the ShardedTensor created
    sharded_tensor._local_shards = local_shards
    # run post initialization, i.e. map registration, rpc initialization
    sharded_tensor._post_init()
    # We need to delete the orig param key if use_orig_param
    keys_to_delete = [key for key in state_dict.keys() for fqn in fqns if fqn in key]
    no_fsdp_prefix = _remove_sub_string(prefix, FSDP_PREFIX)
    state_dict.setdefault(FSDP_FLAT_PARAM_META, {})
    state_dict[FSDP_FLAT_PARAM_META].setdefault(FSDP_FLAT_PARAM_TO_FQNS, {})[f"{no_fsdp_prefix}{FLAT_PARAM}"] = [
        f"{module_name}.{param_name}"
        for param_name, module_name in _module_handle(fsdp_state, module).param_module_names()
    ]
    if tied_weights:
        for key in tied_weights:
            if key in state_dict[FSDP_FLAT_PARAM_META][FSDP_FLAT_PARAM_TO_FQNS][f"{no_fsdp_prefix}{FLAT_PARAM}"]:
                logger.info("drop tied weight %s", key)
                state_dict[FSDP_FLAT_PARAM_META][FSDP_FLAT_PARAM_TO_FQNS][f"{no_fsdp_prefix}{FLAT_PARAM}"].remove(key)
    state_dict[FSDP_FLAT_PARAM_META].setdefault(FSDP_IS_PADDING_MASK, {})[f"{no_fsdp_prefix}{FLAT_PARAM}"] = (
        is_padding_mask
    )
    state_dict[FSDP_FLAT_PARAM_META].setdefault(FSDP_NUMELS_WITH_PADDING, {})[f"{no_fsdp_prefix}{FLAT_PARAM}"] = (
        numels_with_padding
    )
    state_dict[FSDP_FLAT_PARAM_META].setdefault(FSDP_SHAPES, {})[f"{no_fsdp_prefix}{FLAT_PARAM}"] = shapes
    assert len(state_dict[FSDP_FLAT_PARAM_META][FSDP_FLAT_PARAM_TO_FQNS][f"{no_fsdp_prefix}{FLAT_PARAM}"]) == len(
        state_dict[FSDP_FLAT_PARAM_META][FSDP_SHAPES][f"{no_fsdp_prefix}{FLAT_PARAM}"]
    )
    for key in set(keys_to_delete):
        # Hard code now
        if "wg_ema" in key or "ce_ema" in key:
            continue
        else:
            del state_dict[key]
    if tied_weights:
        for key in tied_weights:
            # not saving tied weights
            if key in state_dict:
                logger.info("remove tied weight %s from state dict", key)
                del state_dict[key]
    state_dict[f"{prefix}{FLAT_PARAM}"] = sharded_tensor

    return state_dict


def hack_local_pre_load_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    """
    This hook finds the local flat_param for this FSDP module from the
    state_dict. The flat_param should be a ShardedTensor. This hook converts
    the ShardedTensor to a tensor. No copy happen unless padding is required.
    """
    if fsdp_state._is_root:
        logger.info("into bytecheckpoint hack_local_pre_load_state_dict_hook")
    _lazy_init(fsdp_state, module)
    _replace_by_prefix(state_dict, prefix, f"{prefix}{FSDP_PREFIX}")
    fqn = f"{prefix}{FSDP_PREFIX}{FLAT_PARAM}"
    if fqn not in state_dict:
        assert not _has_fsdp_params(fsdp_state, module), (
            "No `FlatParameter` in `state_dict` for this FSDP instance but it has parameters"
        )
        return
    load_tensor = state_dict[fqn]
    assert isinstance(load_tensor, ShardedTensor), "Tensors in local_state_dict should be ShardedTensor."

    # Convert the ShardedTensor to a Tensor.
    flat_param = _module_handle(fsdp_state, module).flat_param
    assert flat_param is not None
    valid_data_size = flat_param.numel() - flat_param._shard_numel_padded
    shards = load_tensor.local_shards()
    if valid_data_size > 0:
        assert len(shards), "load_local_state_dict assume one shard per ShardedTensor."
        load_tensor = shards[0].tensor
        # Get the metadata of the flat_param to decide whether to pad the loaded
        # tensor.
        if flat_param._shard_numel_padded > 0:
            assert load_tensor.numel() < flat_param.numel(), (
                f"Local shard size = {flat_param.numel()} and the tensor in the state_dict is {load_tensor.numel()}."
            )
            load_tensor = F.pad(load_tensor, [0, flat_param._shard_numel_padded])
    else:
        load_tensor = flat_param
    del state_dict[fqn]
    # We need to use sharded view if use_orig_param == True
    for shard_param_info, fqn in zip(flat_param._shard_param_infos, flat_param._fqns):
        fqn = f"{prefix}{FSDP_PREFIX}{fqn}"
        if not shard_param_info.in_shard:
            # Allow the original data to be freed via garbage collection
            state_dict[fqn] = torch.empty(0)
        else:
            offset = shard_param_info.offset_in_shard
            numel_in_shard = shard_param_info.numel_in_shard
            state_dict[fqn] = load_tensor[offset : offset + numel_in_shard]


@no_type_check
def hack_set_use_dtensor(fsdp_state: _FSDPState) -> None:
    """
    Modify the state dict configuration to use DTensor if the device mesh is provided and the state dict type is not local.

    Args:
        fsdp_state (_FSDPState): The state of the FSDP module.

    This function checks if a device mesh is provided during the initialization of the FSDP module.
    If a device mesh is present and the state dict type is not LOCAL_STATE_DICT, it sets the
    _use_dtensor flag in the state dict configuration to True. If the state dict type is
    LOCAL_STATE_DICT, it suppresses the potential runtime error and does nothing.
    """
    # If device_mesh is passed in when initalizing FSDP, we do not raise runtime error when state dict type is local.
    if getattr(fsdp_state, "_device_mesh", None):
        state_dict_type = fsdp_state._state_dict_type
        if state_dict_type == StateDictType.LOCAL_STATE_DICT:
            # raise RuntimeError(
            #     "Found state_dict_type LOCAL_STATE_DICT",
            #     "DeviceMesh is not compatible with LOCAL_STATE_DICT.",
            #     "Please set state_dict_type to SHARDED_STATE_DICT to get DTensor state_dict.",
            # )
            pass
        else:
            fsdp_state._state_dict_config._use_dtensor = True


@no_type_check
def hack_set_optim_use_dtensor(
    fsdp_state: _FSDPState,
    state_dict_settings: StateDictSettings,
) -> None:
    """
    Modify the optimizer state dict configuration to use DTensor if the device mesh is provided and the state dict type is not local.

    Args:
        fsdp_state (_FSDPState): The state of the FSDP module.
        state_dict_settings (StateDictSettings): The state dict settings of the FSDP module.

    This function checks if a device mesh is provided during the initialization of the FSDP module.
    If a device mesh is present and the state dict type is not LOCAL_STATE_DICT, it sets the
    _use_dtensor flag in the optimizer state dict configuration to True. If the state dict type is
    LOCAL_STATE_DICT, it suppresses the potential runtime error and does nothing.
    """
    # If device_mesh is passed in when initalizing FSDP, we do not raise runtime error
    # when state dict type is local
    if getattr(fsdp_state, "_device_mesh", None):
        state_dict_type = state_dict_settings.state_dict_type
        if state_dict_type == StateDictType.LOCAL_STATE_DICT:
            # raise RuntimeError(
            #     "Found state_dict_type LOCAL_STATE_DICT",
            #     "DeviceMesh is not compatible with LOCAL_STATE_DICT.",
            #     "Please set state_dict_type to SHARDED_STATE_DICT to get DTensor state_dict.",
            # )
            pass
        else:
            state_dict_settings.optim_state_dict_config._use_dtensor = True


def update_local_post_state_dict_hook(tied_weights=None):
    torch.distributed.fsdp._state_dict_utils._local_post_state_dict_hook = partial(
        hack_local_post_state_dict_hook, tied_weights=tied_weights
    )
