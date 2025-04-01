################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
################################################################################

import dataclasses
import os
import pickle
import time
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch.distributed as dist
from torch.distributed.checkpoint.planner import SavePlan

try:
    from torch.distributed.device_mesh import DeviceMesh
except ImportError as e:
    # compatible with torch version <=2.1.0
    from torch.distributed._tensor import DeviceMesh

from bytecheckpoint import BYTECHECKPOINT_GLOBAL_CONFIG
from bytecheckpoint import __version__ as BYTECHECKPOINT_VERSION
from bytecheckpoint.checkpointer.meta_type import (
    STATE_DICT_TYPE,
    Metadata,
    WriteResult,
    _StoragePrefix,
)
from bytecheckpoint.distributed.base_wrapper import _BaseTreeTopoWrapper
from bytecheckpoint.engine import _store_engine
from bytecheckpoint.planner.default_planner import DefaultSavePlanner
from bytecheckpoint.storage import CKPTCounter, write_to_store
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()


FSDP_ENABLED = False
try:
    from bytecheckpoint.distributed import _FSDPTreeTopoWrapper
    from bytecheckpoint.planner.fsdp.fsdp_planner import FSDPSavePlanner

    FSDP_ENABLED = True
except ImportError as e:
    logger.debug("FSDP not found")


def save_state_dict(
    state_dict: STATE_DICT_TYPE,
    state_path: str,
    root_path: str,
    ckpt_name: str,
    framework_name: str,
    suffix: Optional[str],
    planner: DefaultSavePlanner,
    save_ckpt_start_time: float,
    ckpt_counter: CKPTCounter,
    global_steps: Optional[int],
    callback: Optional[Callable],
    pg_or_mesh: Optional[Union[dist.ProcessGroup, DeviceMesh]] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    async_io: bool = True,
) -> None:
    """
    Saves a distributed ``state_dict`` in SPMD style.

    Args:
        state_dict (STATE_DICT_TYPE): The state dictionary to be saved.
        state_path (str): The local path where the checkpoint data will be written.
        root_path (str): The root path for the checkpoint storage.
        ckpt_name (str): The name of the checkpoint.
        framework_name (str): The name of the framework used.
        suffix (Optional[str]): An optional suffix for the checkpoint.
        planner (DefaultSavePlanner): The planner used for saving the checkpoint.
        save_ckpt_start_time (float): The start time of the checkpoint saving process.
        ckpt_counter (CKPTCounter): The counter for the checkpoint.
        global_steps (Optional[int]): The global steps of the checkpoint.
        callback (Optional[Callable]): An optional callback function.
        pg_or_mesh (Optional[Union[dist.ProcessGroup, DeviceMesh]]): The process group or device mesh. Defaults to None.
        coordinator_rank (int): The rank of the coordinator. Defaults to 0.
        no_dist (bool): Whether to disable distributed communication. Defaults to False.
        async_io (bool): Whether to use asynchronous I/O. Defaults to True.

    Returns:
        None: This function does not return anything.
    """

    # Check save planner.
    assert can_support_save(planner), "Unsupported planner for saving checkpoints."
    # Create distributed world based on process group and coordinator rank.
    distW = setup_save_distW(planner, coordinator_rank, pg_or_mesh, no_dist)
    # Lookup plan and metadata cache.
    cached_data = planner.lookup_plan_meta(extra_info=suffix)

    # Step 1: all processes create local write plan,
    # then coordinator gathers all local plans and create global plan.

    # Setup planner and global metadata.
    assert planner is not None
    planner.set_up_planner(state_dict, distW.is_coordinator)
    global_metatadata = None

    # Define local and global steps.
    def local_step():
        local_step_start_time = time.time()
        logger.debug("[Rank = %s] Start local step of planning", dist.get_rank())
        local_plan = planner.create_local_plan()
        local_plan = prepare_local_plan(local_plan)
        local_step_time_cost = time.time() - local_step_start_time
        logger.debug("[Rank = %s] Finish local step of planning. Time: %s s", dist.get_rank(), local_step_time_cost)
        return local_plan

    def global_step(all_local_plans: List[SavePlan]):
        global_step_start_time = time.time()
        logger.debug("[Rank = %s] Start global step of planning", dist.get_rank())
        nonlocal global_metatadata
        assert planner is not None
        all_local_plans, global_metatadata = planner.create_global_plan(all_local_plans)
        all_local_plans = prepare_global_plan(all_local_plans)
        global_step_time_cost = time.time() - global_step_start_time
        logger.debug("[Rank = %s] End global step of planning. Time cost: %s s", dist.get_rank(), global_step_time_cost)
        return all_local_plans

    # Execute the planning protocol.
    # Each worker bypass the `reduce_scatter()` if finding cached central_plan.
    # NOTE: it fails when the plans of partial workers change while others keep the same.
    logger.info("[Rank = %s] Start planning.", dist.get_rank())
    plan_start_time = time.time()
    if cached_data:
        logger.debug("Rank=%s Plan cache hit. Reuse existing plan", dist.get_rank())
        central_plan, _ = cached_data
        prepare_local_plan(central_plan)
    else:
        logger.debug("Rank=%s Plan cache miss. The model/optimizer appears for the first time.", dist.get_rank())
        central_plan = distW.exec_reduce_scatter("plan", local_step, global_step)
    plan_cost_time = time.time() - plan_start_time
    logger.info("[Rank = %s] Finish planning. Time cost: %s s", dist.get_rank(), plan_cost_time)

    # Step 2: all processes write data from GPUs to pinned memory pool, then dump to local path
    # then coordinator write meta-data to local path.

    # Get write futures when cache missing.
    write_futures = []
    write_results = []

    # Define local and global steps.
    def write_data(async_io: bool = False):
        # get the final local plan.
        nonlocal write_results
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)

        # Register resources lazily.
        assert _store_engine.register_resources(ckpt_name, framework_name, suffix, state_path)
        # Use pinned memory pool and mult_processing for dumping ckpt to local directory efficiently.
        returned_write_futures = _store_engine.execute(
            ckpt_name,
            framework_name,
            suffix,
            final_local_plan,
            planner,
            async_io,
        )
        write_results = returned_write_futures
        if async_io:
            return returned_write_futures
        else:
            values = []
            for fut in returned_write_futures:
                curr_val = fut.result()
                values += curr_val
            return values

    def finish_checkpoint(all_results: List[List[WriteResult]]):
        logger.debug("Start writing metadata")
        assert global_metatadata is not None, f"rank: {distW.get_rank()} has no global_metadata"
        path = _store_engine.get_path(ckpt_name, framework_name, suffix)
        finish(path=path, metadata=global_metatadata, results=all_results)
        logger.debug("Finish writing metadata")
        return global_metatadata

    # Each worker bypass the `all_reduce()` if finding cached metadata.
    # NOTE: it fails when the plans of partial workers change while others keep the same.
    logger.info("[Rank = %s] Start local file writing and gather metadata", dist.get_rank())
    store_local_start_time = time.time()
    if cached_data:
        logger.debug("Metdata cache hit. Reuse existing metadata")
        _, final_storage_metadata = cached_data
        write_results = write_data(async_io=async_io)
        # Be sure to write cache metadata to .metadata file
        # Otherwises only the first checkpoint has .metadata
        # which leads to error when loading other checkpoints
        if distW.is_coordinator:
            path = _store_engine.get_path(ckpt_name, framework_name, suffix)
            with (state_path / ".metadata.tmp").open("wb") as metadata_file:
                pickle.dump(final_storage_metadata, metadata_file)
                os.fsync(metadata_file.fileno())

            (path / ".metadata.tmp").rename(path / ".metadata")

        if async_io:
            write_futures = write_results
    else:
        logger.debug("Rank=%s Metadata cache miss. The model/optimizer appears for the first time.", dist.get_rank())
        # First time do synchronous storing to get final_storage_metatdata.
        final_storage_metadata = distW.exec_all_reduce("write", write_data, finish_checkpoint)
        if async_io:
            write_futures = write_results
        assert central_plan is not None
        assert final_storage_metadata is not None
        planner.cache_plan_meta(central_plan, final_storage_metadata, extra_info=suffix)
    store_local_cost_time = time.time() - store_local_start_time
    logger.info(
        "[Rank = %s] Finish local file writing and gather metadata. Time cost: %s s",
        dist.get_rank(),
        store_local_cost_time,
    )

    # Write checkpoint to storage.
    write_to_store(
        ckpt_name=ckpt_name,
        framework_name=framework_name,
        suffix=suffix if suffix else "",
        fast_saving=async_io,
        ckpt_path=root_path,
        save_ckpt_start_time=save_ckpt_start_time,
        local_write_futures={state_path: write_futures},
        ckpt_counter=ckpt_counter,
        global_steps=global_steps,
        callback=callback,
    )


"""
Planning Methods.
"""


def prepare_local_plan(plan: SavePlan) -> SavePlan:
    return plan


def prepare_global_plan(global_plan: List[SavePlan]) -> List[SavePlan]:
    new_plans = [
        dataclasses.replace(plan, storage_data=_StoragePrefix(f"__{i}_")) for i, plan in enumerate(global_plan)
    ]
    return new_plans


def finish(
    path: Path,
    metadata: Metadata,
    results: List[List[WriteResult]],
) -> None:
    all_byte_md = dict()
    all_tensor_md = dict()
    # Update all byte metadata and tensor metadata.
    for wr_list in results:
        all_byte_md.update({wr.index: wr.byte_metadata for wr in wr_list})
        all_tensor_md.update({wr.index: wr.tensor_metadata for wr in wr_list})
    # Add version information.
    metadata.bytecheckpoint_version = BYTECHECKPOINT_VERSION
    metadata.storage_data = all_byte_md
    metadata.all_tensor_metadata = all_tensor_md
    # Write the metadata file.
    with (path / ".metadata.tmp").open("wb") as metadata_file:
        pickle.dump(metadata, metadata_file)
        os.fsync(metadata_file.fileno())
    (path / ".metadata.tmp").rename(path / ".metadata")


"""
Helper Methods.
"""


def can_support_save(planner: DefaultSavePlanner) -> bool:
    return FSDP_ENABLED and isinstance(planner, FSDPSavePlanner)


def setup_save_distW(
    planner: DefaultSavePlanner,
    coordinator_rank: int,
    pg_or_mesh: Optional[Union[dist.ProcessGroup, DeviceMesh]],
    no_dist: bool,
) -> _BaseTreeTopoWrapper:
    """
    Setup a distributed communication wrapper for saving checkpoints.

    Args:
        planner (DefaultSavePlanner): The planner used for saving the checkpoint.
        coordinator_rank (int): The rank of the coordinator.
        pg_or_mesh (Optional[Union[dist.ProcessGroup, DeviceMesh]]): The process group or device mesh.
        no_dist (bool): Whether to disable distributed communication.

    Returns:
        _BaseTreeTopoWrapper: A wrapper for distributed communication.
    """
    distW = None
    # Setup distributed communication wrapper,
    if FSDP_ENABLED and isinstance(planner, FSDPSavePlanner):
        assert pg_or_mesh is None or isinstance(pg_or_mesh, dist.ProcessGroup)
        distW = _FSDPTreeTopoWrapper(
            pg_or_mesh,
            not no_dist,
            coordinator_rank,
            BYTECHECKPOINT_GLOBAL_CONFIG.enable_tree_topo,
        )
    else:
        raise ValueError("Unsupported planner for planning")
    # Set coordinator rank.
    if isinstance(pg_or_mesh, DeviceMesh):
        distW.coordinator_rank = dist.get_global_rank(pg_or_mesh._get_or_create_default_group(), distW.coordinator_rank)
    elif isinstance(pg_or_mesh, dist.ProcessGroup):
        distW.coordinator_rank = dist.get_global_rank(pg_or_mesh, distW.coordinator_rank)
    return distW
