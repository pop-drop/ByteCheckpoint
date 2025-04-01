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
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Union

import torch.distributed as dist
from torch.distributed.checkpoint.planner import LoadPlan, ReadItem

try:
    from torch.distributed.device_mesh import DeviceMesh
except ImportError as e:
    # compatible with torch version <=2.1.0
    from torch.distributed._tensor import DeviceMesh

from bytecheckpoint.checkpointer.meta_type import STATE_DICT_TYPE
from bytecheckpoint.distributed.base_wrapper import _BaseTreeTopoWrapper
from bytecheckpoint.engine import _load_engine
from bytecheckpoint.planner.default_planner import DefaultLoadPlanner
from bytecheckpoint.storage import read_from_store
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()

META_DATA_FILE = ".metadata"

FSDP_ENABLED = False
try:
    from bytecheckpoint.distributed import _FSDPTreeTopoWrapper
    from bytecheckpoint.planner.fsdp.fsdp_planner import FSDPLoadPlanner

    FSDP_ENABLED = True
except ImportError as e:
    logger.debug("FSDP not found")


from bytecheckpoint import BYTECHECKPOINT_GLOBAL_CONFIG


def load_state_dict(
    state_dict: STATE_DICT_TYPE,
    path: str,
    ckpt_name: str,
    framework_name: str,
    suffix: Optional[str],
    planner: Optional[DefaultLoadPlanner],
    planning_pg_or_mesh: Optional[Union[dist.ProcessGroup, DeviceMesh]] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    fast_loading: bool = False,
) -> List[Future[Any]]:
    """
    Loads a distributed ``state_dict`` in SPMD style.

    Args:
        state_dict (STATE_DICT_TYPE): The state dictionary to be loaded.
        path (str): The path where the checkpoint is stored.
        ckpt_name (str): The name of the checkpoint.
        framework_name (str): The name of the framework.
        suffix (Optional[str]): An optional suffix for the checkpoint.
        planner (Optional[DefaultLoadPlanner]): The load planner to be used.
        planning_pg_or_mesh (Optional[Union[dist.ProcessGroup, DeviceMesh]]): The process group or device mesh for planning. Defaults to None.
        coordinator_rank (int): The rank of the coordinator. Defaults to 0.
        no_dist (bool): Whether to disable distributed training. Defaults to False.
        fast_loading (bool): Whether to enable parallel loading. Defaults to False.

    Returns:
        List[Future[Any]]: A list of futures representing the asynchronous loading operations.
    """

    # Check load planner.
    assert can_support_load(planner), "Unsupported planner for loading checkpoints."
    # Create distributed world based on process group and coordinator rank
    # NOTE: we do global planning across all ranks by default.
    distW = setup_load_distW(planner, coordinator_rank, planning_pg_or_mesh, no_dist)

    # Step 1: read metadata in sync mode.
    meta_read_start_time = time.time()
    metadata_future = read_from_store(
        ckpt_name,
        framework_name,
        suffix,
        [META_DATA_FILE],
        path,
        async_io=False,
    )
    assert len(metadata_future) == 1, f"Only one metadata file is expected for loading the checkpoint of {ckpt_name}"
    _, metadata_path = metadata_future[0].result()
    # Register the resources lazily.
    assert _load_engine.register_resources(ckpt_name, framework_name, suffix)
    metadata = _load_engine.load_metadata(metadata_path)
    meta_read_cost_time = time.time() - meta_read_start_time
    logger.debug("Finish read meta file. Cost time: %s s", meta_read_cost_time)

    # Step 2: all processes create local read final_local_plan,
    # then coordinator gathers all local plans and create global final_local_plan.

    # Setup planner.
    assert planner is not None
    planner.set_up_planner(state_dict, metadata, distW.is_coordinator)

    # Define local and global steps.
    def local_step():
        local_step_start_time = time.time()
        logger.debug("[Rank = %s] Start local step of planning", dist.get_rank())
        local_plan = planner.create_local_plan()
        local_plan = prepare_local_plan(local_plan)
        local_step_time_cost = time.time() - local_step_start_time
        logger.debug("[Rank = %s] Finish local step of planning. Time: %s s", dist.get_rank(), local_step_time_cost)
        # TODO: support read_comm_overlap.
        # Set `dp_global_ranks` if enable read_comm_overlap.
        # if read_comm_overlap:
        #     assert storage_reader.dp_global_ranks is not None
        #     local_plan.dp_global_ranks = storage_reader.dp_global_ranks
        return local_plan

    def global_step(all_local_plans: List[LoadPlan]):
        assert planner is not None
        all_local_plans = planner.create_global_plan(all_local_plans)
        all_local_plans = prepare_global_plan(all_local_plans)
        return all_local_plans

    # Execute the planning protocol.
    # NOTE: global planning and reduce scatter communication is not necessary.
    logger.info("[Rank = %s] Start planning.", dist.get_rank())
    plan_start_time = time.time()
    central_plan = distW.exec_reduce_scatter("plan", local_step, global_step)
    load_ckpt_plan_cost_time = time.time() - plan_start_time
    logger.info("[Rank = %s] Finish planning. Time cost: %s s", dist.get_rank(), load_ckpt_plan_cost_time)

    # Step 3: all processes read data from HDFS path or local path, then load data into the sate_dict.

    # Define the local step.
    def read_data(fast_loading: bool = False) -> List[Future[Any]]:
        # Get the final local plan.
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)

        # Divide read items into read_items and comm_items.
        read_items = [read_item for read_item in final_local_plan.items if isinstance(read_item, ReadItem)]
        # Group read requests by file.
        relative_path_to_read_items: Dict[str, List[ReadItem]] = dict()
        for read_item in read_items:
            item_md = metadata.storage_data[read_item.storage_index]
            relative_path_to_read_items.setdefault(item_md.relative_path, []).append(read_item)

        # Read from storage.
        relative_paths = list(relative_path_to_read_items.keys())
        read_futures = read_from_store(
            ckpt_name,
            framework_name,
            suffix,
            relative_paths,
            path,
            async_io=True,
        )
        # Load from file.
        load_futures = _load_engine.execute(
            ckpt_name,
            framework_name,
            suffix,
            relative_path_to_read_items,
            read_futures,
            planner,
            metadata,
            fast_loading=fast_loading,
        )
        return load_futures

    read_start_time = time.time()
    load_futures = read_data(fast_loading=fast_loading)
    dist.barrier()

    read_cost_time = time.time() - read_start_time
    logger.info("[Rank = %s] Finish reading. Time cost: %s s", dist.get_rank(), read_cost_time)

    return load_futures


"""
Planning Methods.
"""


def prepare_local_plan(plan: LoadPlan) -> LoadPlan:
    return plan


def prepare_global_plan(global_plan: List[LoadPlan]) -> List[LoadPlan]:
    return global_plan


"""
Helper Methods.
"""


def can_support_load(planner: DefaultLoadPlanner) -> bool:
    return FSDP_ENABLED and isinstance(planner, FSDPLoadPlanner)


def setup_load_distW(
    planner: DefaultLoadPlanner,
    coordinator_rank: int,
    pg_or_mesh: Optional[Union[dist.ProcessGroup, DeviceMesh]],
    no_dist: bool,
) -> _BaseTreeTopoWrapper:
    """
    Sets up the distributed world for loading a checkpoint.

    Args:
        planner (DefaultLoadPlanner): The load planner to be used.
        coordinator_rank (int): The rank of the coordinator.
        pg_or_mesh (Optional[Union[dist.ProcessGroup, DeviceMesh]]): The process group or device mesh for planning.
        no_dist (bool): Whether to disable distributed training.

    Returns:
        _BaseTreeTopoWrapper: A wrapper representing the distributed world.
    """
    distW = None
    if FSDP_ENABLED and isinstance(planner, FSDPLoadPlanner):
        assert pg_or_mesh is None or isinstance(pg_or_mesh, dist.ProcessGroup)
        distW = _FSDPTreeTopoWrapper(
            pg_or_mesh,
            not no_dist,
            coordinator_rank,
            BYTECHECKPOINT_GLOBAL_CONFIG.enable_tree_topo,
        )
    else:
        raise ValueError("Unsupported planner for planning")
    if isinstance(pg_or_mesh, DeviceMesh):
        distW.coordinator_rank = dist.get_global_rank(pg_or_mesh._get_or_create_default_group(), distW.coordinator_rank)
    elif isinstance(pg_or_mesh, dist.ProcessGroup):
        distW.coordinator_rank = dist.get_global_rank(pg_or_mesh, distW.coordinator_rank)
    return distW
