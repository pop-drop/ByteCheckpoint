################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
################################################################################
import time
from typing import TypeVar

import torch.distributed as dist

from ..utilities.logger import get_bytecheckpoint_logger
from .base_wrapper import DP_LEVEL, INTRA_NODE_DP_SIZE, LEVEL_TREE_CACHE, _BaseTreeTopoWrapper

logger = get_bytecheckpoint_logger()

T = TypeVar("T")
R = TypeVar("R")


class _FSDPTreeTopoWrapper(_BaseTreeTopoWrapper):
    """
    A TreeTopoWrapper class for FSDP training framework.
    """

    def construct_topo(self):
        """
        Construct the communication tree topology through ``torch.distributed``.

        This method constructs a two-level communication tree topology for data parallelism.
        It first checks if the world size is divisible by the intra-node data parallel size.
        If so, it tries to retrieve the level tree information from the cache. If the cache is empty,
        it constructs new level trees, including level-1 groups within nodes and a level-2 group across nodes,
        and then caches the constructed level trees.

        Returns:
            None: This method modifies the instance variables `level_trees` and `is_constructed` in-place.
        """
        self.level = DP_LEVEL
        self.level_trees = [None] * self.level
        construct_start_rime = time.time()
        if dist.get_world_size() % INTRA_NODE_DP_SIZE == 0:
            # Look up the level tree cache.
            if LEVEL_TREE_CACHE["DP_LEVEL"] is not None:
                self.level_trees = LEVEL_TREE_CACHE["DP_LEVEL"]
            # Build new level trees and cache them.
            else:
                # Construct level-1 group.
                level1_root_ranks = list(range(0, dist.get_world_size(), INTRA_NODE_DP_SIZE))
                level1_rank_lists = [
                    list(range(root_rank, root_rank + INTRA_NODE_DP_SIZE)) for root_rank in level1_root_ranks
                ]
                my_root_rank = (dist.get_rank() // INTRA_NODE_DP_SIZE) * INTRA_NODE_DP_SIZE
                for root_rank, rank_list in zip(level1_root_ranks, level1_rank_lists):
                    group = dist.new_group(ranks=rank_list)
                    if my_root_rank == root_rank:
                        self.level_trees[0] = (root_rank, group)
                # Construct level-2 group.
                group = dist.new_group(ranks=level1_root_ranks)
                if dist.get_rank() == my_root_rank:
                    self.level_trees[1] = (0, group)
                # Cache the level trees.
                self.cache_level_trees("DP_LEVEL")
            self.is_constructed = True
        else:
            logger.warning(
                "For data parallelism, the world size is not divisible by %s, fail to use tree topology",
                INTRA_NODE_DP_SIZE,
            )
        construct_time = time.time() - construct_start_rime
        logger.info("Finish tree topo construction. Time cost: %s s", construct_time)
