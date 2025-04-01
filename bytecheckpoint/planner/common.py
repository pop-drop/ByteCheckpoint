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

import collections
import dataclasses
from collections import OrderedDict
from typing import Dict, Hashable, List, Optional, Tuple

import torch
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import SavePlan, WriteItem, WriteItemType

from bytecheckpoint.checkpointer.meta_type import Metadata
from bytecheckpoint.config import BYTECHECKPOINT_GLOBAL_CONFIG

from ..utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()


_MAX_CACHE_SIZE = BYTECHECKPOINT_GLOBAL_CONFIG.planner_lru_cache_capacity


class PlanLRUCache:
    def __init__(self) -> None:
        self._cache: OrderedDict[Hashable, Tuple[SavePlan, Metadata]] = OrderedDict()
        self._capacity = _MAX_CACHE_SIZE

    def get(self, key: Hashable) -> Optional[Tuple[SavePlan, Metadata]]:
        if key in self._cache:
            return self._cache[key]
        else:
            return None

    def put(self, key: Hashable, plan_value: SavePlan, metadata_value: Metadata) -> None:
        if key in self._cache:
            self._cache.move_to_end(key, last=False)
        else:
            self._cache[key] = (plan_value, metadata_value)
            if len(self._cache) > self._capacity:
                self._cache.popitem()

    def clear(self) -> None:
        self._cache.clear()
        self._capacity = _MAX_CACHE_SIZE

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return f"PlanLURCache(capacity: {self._capacity}, keys: {tuple(self._cache.keys())})"


GLOBAL_PLAN_CACHE = PlanLRUCache()


def custom_dedup_tensors(all_plans: List[SavePlan]) -> List[SavePlan]:
    """
    A function to remove duplicate tensors to write
    when creating global writing plan for saving checkpoint
    During the deduplication,
    we balance the workloads for duplicated tensors

    Args:
        all_plans (List[SavePlan]): A list of SavePlan objects representing all the plans to be processed.

    Returns:
        List[SavePlan]: A list of SavePlan objects with duplicate tensors removed and workload balanced.
    """
    key_to_plan: Dict[MetadataIndex, List[int]] = {}
    key_to_write_item: Dict[MetadataIndex, WriteItem] = {}

    # For all ranks, Build mapping between MetadataIndex and rank with write item
    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            key_to_plan.setdefault(write_item.index, []).append(plan_idx)
            key_to_write_item[write_item.index] = write_item

    # Find out duplicated items
    replicated_items = {k: v for k, v in key_to_plan.items() if len(v) > 1}
    # Remove duplicates by always keeping the first entry (Not balance).
    # Compute the per-rank remove set.
    plan_to_keys: Dict[int, List[MetadataIndex]] = {}
    # Record the size of non-duplicated tensors assigned to each rank
    assigned_work_load = collections.defaultdict(int)
    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            if write_item.index not in replicated_items:
                # tensor_data present if it's a tensor write
                if write_item.tensor_data:
                    assigned_work_load[plan_idx] += write_item.tensor_data.size.numel()
                else:
                    assigned_work_load[plan_idx] += 1

    # Sort tensors in ascending order
    sorted_keys = sorted(
        replicated_items.keys(),
        key=lambda k: -key_to_write_item[k].tensor_data.size.numel() if key_to_write_item[k].tensor_data else -1,
    )

    for key in sorted_keys:
        plans = replicated_items[key]
        # For duplicated tensors, select the rank assigned with minimum tensor size so far
        write_item = key_to_write_item[key]
        # if the tensor is a scalar such as optimizer step, we assign it to the first writer
        if write_item.tensor_data and write_item.type == WriteItemType.TENSOR:
            writer_id = min(plans)
        else:
            writer_id = min(plans, key=lambda k: assigned_work_load[k])

        # tensor_data present if it's a tensor write
        if write_item.tensor_data:
            assigned_work_load[writer_id] += write_item.tensor_data.size.numel()
        else:
            assigned_work_load[writer_id] += 1

        for plan_idx in plans:
            # If the rank is not writer rank, remove the key in the rank's plan
            if plan_idx != writer_id:
                plan_to_keys.setdefault(plan_idx, []).append(key)
    logger.debug("Duplicate keys to remove: %s", plan_to_keys)

    for plan_idx, keys in plan_to_keys.items():
        # Key Set contains keys to remove
        key_set = set(keys)
        # rewrite items and remove elements
        new_items = [write_item for write_item in all_plans[plan_idx].items if write_item.index not in key_set]
        all_plans[plan_idx] = dataclasses.replace(all_plans[plan_idx], items=new_items)

    return all_plans


def _init_optim_state(optim: torch.optim.Optimizer) -> None:
    """
    Initialize optim states by calling the step() with zero grads.
    """
    if not hasattr(optim, "state"):
        if hasattr(optim, "optimizer"):
            _init_optim_state(optim.optimizer)
        return

    if optim.state:
        # The optimizer state is initialized.
        return

    for param_group in optim.param_groups:
        for param in param_group["params"]:
            if param.grad is not None:
                raise RuntimeError(
                    "state_dict can only be used if the optimizer states are initialized (usually after one step() "
                    "with gradients) or gradients are None. For the later case, state_dict will fake the gradients "
                    "as zero to initialize the optimizer states. However, the gradients are not None. This case often "
                    "occurs when using a single main model with multiple optimizers where an optimizer's state is not "
                    "initialized but the main model do have gradients. Please ensure that all optimizers have called "
                    "zero_grad."
                )
            if param.requires_grad and param.size() != torch.Size([0]):
                param.grad = torch.zeros_like(param)
    optim.step(closure=None)

    if torch.__version__ >= "2":
        optim.zero_grad()
    else:
        optim.zero_grad(set_to_none=True)
