################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import Optional, Tuple

from torch.distributed.checkpoint.planner import SavePlan

from bytecheckpoint.checkpointer.meta_type import STATE_DICT_STR, Metadata
from bytecheckpoint.planner.common import GLOBAL_PLAN_CACHE
from bytecheckpoint.planner.default_planner import create_plan_cache_key
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

from ..fsdp.fsdp_planner import FSDPLoadPlanner, FSDPSavePlanner

logger = get_bytecheckpoint_logger()
__all__ = [
    "DDPSavePlanner",
    "DDPLoadPlanner",
]


class DDPLoadPlanner(FSDPLoadPlanner):
    """
    A planner class for loading DDP checkpoint using PyTorch DCP
    """

    def __init__(self, strict: bool):
        super().__init__(strict)


class DDPSavePlanner(FSDPSavePlanner):
    """
    A planner class for saving DDP checkpoint using PyTorch DCP
    """

    def __init__(self):
        super().__init__()

    def lookup_plan_meta(self, extra_info: Optional[str] = "") -> Optional[Tuple[SavePlan, Metadata]]:
        if not hasattr(self, STATE_DICT_STR):
            return None
        else:
            plan_key = create_plan_cache_key(self.state_dict, extra_info)
            return GLOBAL_PLAN_CACHE.get(plan_key)

    def cache_plan_meta(self, new_plan: SavePlan, new_metadata: Metadata, extra_info: Optional[str] = "") -> None:
        plan_key = create_plan_cache_key(self.state_dict, extra_info)
        GLOBAL_PLAN_CACHE.put(plan_key, new_plan, new_metadata)
