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
import io
import logging
import operator
import pickle
import sys
from collections import ChainMap
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint._dedup_tensors import dedup_tensors
from torch.distributed.checkpoint._nested_dict import (
    FLATTEN_MAPPING,
    flatten_state_dict,
)
from torch.distributed.checkpoint._sharded_tensor_utils import (
    _flatten_sharded_tensors,
)
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed.checkpoint.metadata import (
    STATE_DICT_TYPE,
    STORAGE_TYPES,
    BytesStorageMetadata,
    ChunkStorageMetadata,
    MetadataIndex,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.planner_helpers import (
    _create_default_metadata_only_plan,
    _create_read_items,
    _create_write_items,
)
from torch.distributed.checkpoint.utils import find_state_dict_object

from bytecheckpoint.checkpointer.meta_type import Metadata

logger: logging.Logger = logging.getLogger(__file__)

__all__ = [
    "DefaultSavePlanner",
    "DefaultLoadPlanner",
    "create_default_local_load_plan",
    "create_default_global_load_plan",
    "create_default_local_save_plan",
    "create_default_global_save_plan",
]


@dataclass
class PlanCacheKey:
    key_set: Set[str] = field(default_factory=set)
    extra_info: Optional[str] = ""

    def __hash__(self):
        combined = (frozenset(self.key_set), self.extra_info)
        return hash(combined)

    def __eq__(self, other):
        if not isinstance(other, PlanCacheKey):
            return False
        return self.key_set == other.key_set


def create_plan_cache_key(state_dict: Dict[str, Any], extra_info: Optional[str] = "") -> PlanCacheKey:
    """
    Generate a `PlanCacheKey` object based on the keys of a flattened dictionary and optional extra information.

    Args:
        state_dict (Dict[str, Any]): A flattened dictionary where keys are strings and values can be of any type.
        extra_info (Optional[str], optional): Additional information to be included in the cache key. Defaults to an empty string.

    Returns:
        PlanCacheKey: A `PlanCacheKey` object containing a set of keys (without the last part after the last dot) from the input dictionary and the optional extra information.
    """
    plan_cache_key = PlanCacheKey(extra_info=extra_info)
    for key in state_dict:
        plan_cache_key.key_set.add(key.rsplit(".", 1)[0])
    return plan_cache_key


def get_pickled_object_size(obj: object):
    """
    Calculate the size of a pickled object in bytes.

    This function serializes the given object using the `pickle` module and then
    determines the size of the serialized object in bytes.

    Args:
        obj (object): The object to be pickled and measured.

    Returns:
        int: The size of the pickled object in bytes.
    """
    # Serialize the object using pickle.
    pickled_obj = pickle.dumps(obj)

    # Get the size of the serialized object.
    size_in_bytes = sys.getsizeof(pickled_obj)

    return size_in_bytes


class DefaultSavePlanner(SavePlanner):
    mappings: FLATTEN_MAPPING

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        dedup_replicated_tensors: bool = True,
    ) -> None:
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.dedup_replicated_tensors = dedup_replicated_tensors
        self.mappings = {}

    def set_up_planner(self, state_dict: STATE_DICT_TYPE, is_coordinator: bool) -> None:
        """
        Initialize the planner with the given state dictionary and coordinator status.

        This method prepares the planner by potentially flattening the state dictionary
        and sharded tensors, and then stores the processed state dictionary and coordinator status.

        Args:
            state_dict (STATE_DICT_TYPE): The state dictionary to be processed.
            is_coordinator (bool): A flag indicating whether the current process is the coordinator.
        """
        if self.flatten_state_dict:
            state_dict, self.mappings = flatten_state_dict(state_dict)
        if self.flatten_sharded_tensors:
            state_dict = _flatten_sharded_tensors(state_dict)
        self.state_dict = state_dict
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> SavePlan:
        """
        Create a local save plan based on the current state dictionary and coordinator status.

        This method first generates a default local save plan using the provided state dictionary
        and coordinator status. If the `flatten_state_dict` flag is set to True, it updates the
        plan with the flattened mappings. Finally, it stores the plan in the instance variable
        `self.plan` and returns it.

        Returns:
            SavePlan: The generated local save plan.
        """
        plan = create_default_local_save_plan(self.state_dict, self.is_coordinator)
        if self.flatten_state_dict:
            plan = dataclasses.replace(plan, planner_data=self.mappings)
        self.plan = plan

        return self.plan

    def create_global_plan(
        self,
        all_plans: List[SavePlan],
        do_validate_global_plan: bool = True,
    ) -> Tuple[List[SavePlan], Metadata]:
        """
        Create a global save plan and associated metadata from a list of local save plans.

        This method first deduplicates replicated tensors if the `dedup_replicated_tensors` flag is set to True.
        It then creates a default global save plan using the provided local plans. If the `flatten_state_dict` flag
        is set to True, it merges the planner data from all plans and updates the metadata with the merged mappings.
        Finally, it validates the global plan if the `do_validate_global_plan` flag is set to True.

        Args:
            all_plans (List[SavePlan]): A list of local save plans.
            do_validate_global_plan (bool, optional): A flag indicating whether to validate the global plan. Defaults to True.

        Returns:
            Tuple[List[SavePlan], Metadata]: A tuple containing the global save plan and the associated metadata.
        """
        if self.dedup_replicated_tensors:
            all_plans = dedup_tensors(all_plans)

        global_plan, metadata = create_default_global_save_plan(all_plans)

        if self.flatten_state_dict:
            # | does not work for Python 3.8 or older version.
            # merged_mappings = reduce(
            #     lambda x, y: x | y, (p.planner_data for p in global_plan)
            # )
            planner_data_dict = [p.planner_data for p in global_plan]
            merged_mappings = dict(ChainMap(*planner_data_dict))
            metadata = dataclasses.replace(metadata, planner_data=merged_mappings)

        if do_validate_global_plan and not _validate_global_plan(global_plan, metadata):
            raise ValueError("Failed to validate global plan")

        self.global_plan = global_plan
        self.metadata = metadata

        return self.global_plan, self.metadata

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        self.plan = new_plan
        return new_plan

    def resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
        object = self.lookup_object(write_item.index)
        return self.transform_object(write_item, object)

    def lookup_object(self, index: MetadataIndex) -> Any:
        """
        This is an extension from the planner interface to make it easy to extend the default planner.
        """
        return find_state_dict_object(self.state_dict, index)

    def transform_object(self, write_item: WriteItem, object: Any):
        """
        This is an extension from the planner interface to make it easy to extend the default planner.
        """
        if write_item.type == WriteItemType.BYTE_IO:
            byte_buffer = io.BytesIO()
            torch.save(object, byte_buffer)
            object = byte_buffer
        return object

    def lookup_plan_meta(self, extra_info: Optional[str] = "") -> Optional[Tuple[SavePlan, Metadata]]:
        raise NotImplementedError

    def cache_plan_meta(self, plan: SavePlan, metadata: Metadata, extra_info: Optional[str] = "") -> None:
        raise NotImplementedError


class DefaultLoadPlanner(LoadPlanner):
    """
    DefaultLoadPlanner that adds multiple features on top of LoadPlanner.

    In particular it adds the following:

    flatten_state_dict: Handle state_dict with nested dicts
    flatten_sharded_tensors: For FSDP in 2D parallel mode
    """

    original_state_dict: STATE_DICT_TYPE
    mappings: FLATTEN_MAPPING

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        strict: bool = True,
    ) -> None:
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.original_state_dict = {}
        self.mappings = {}
        self.strict = strict

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata,
        is_coordinator: bool,
    ) -> None:
        """
        Initialize the planner with the given state dictionary, metadata, and coordinator status.

        This method prepares the planner by potentially flattening the state dictionary
        and sharded tensors, and then stores the processed state dictionary, metadata, and coordinator status.

        Args:
            state_dict (STATE_DICT_TYPE): The state dictionary to be processed.
            metadata (Metadata): The metadata associated with the state dictionary.
            is_coordinator (bool): A flag indicating whether the current process is the coordinator.
        """
        self.original_state_dict = state_dict

        if self.flatten_sharded_tensors:
            state_dict = _flatten_sharded_tensors(state_dict)

        if self.flatten_state_dict:
            state_dict, self.mappings = flatten_state_dict(state_dict)

        self.state_dict = state_dict
        self.metadata = metadata
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> LoadPlan:
        return create_default_local_load_plan(self.state_dict, self.metadata, self.strict)

    def create_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        return create_default_global_load_plan(global_plan)

    def finish_plan(self, new_plan: LoadPlan) -> LoadPlan:
        return new_plan

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        """
        Load bytes from a given `io.BytesIO` object into the state dictionary.

        This method is responsible for loading the bytes stored in the `value` object into the appropriate
        location in the state dictionary. If the `flatten_state_dict` flag is set to True, it uses the
        mapping to place the loaded object in the original state dictionary. Otherwise, it directly places
        the loaded object in the current state dictionary.

        Args:
            read_item (ReadItem): An object containing information about the destination index in the state dictionary.
            value (io.BytesIO): A byte buffer containing the data to be loaded.

        Returns:
            None
        """
        if self.flatten_state_dict:
            set_element(
                self.original_state_dict,
                self.mappings[read_item.dest_index.fqn],
                torch.load(value),
            )
        else:
            self.state_dict[read_item.dest_index.fqn] = torch.load(value)

    def resolve_tensor(self, read_item: ReadItem):
        tensor = self.lookup_tensor(read_item.dest_index)
        return self.transform_tensor(read_item, tensor)

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        pass

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        """
        This is an extension from the planner interface to make it easy to extend the default planner.
        """
        return find_state_dict_object(self.state_dict, index)

    def transform_tensor(self, read_item: ReadItem, tensor: torch.Tensor):
        """
        This is an extension from the planner interface to make it easy to extend the default planner.
        """
        return narrow_tensor_by_index(tensor, read_item.dest_offsets, read_item.lengths)


def create_default_local_load_plan(state_dict: Dict[str, Any], metadata: Metadata, strict: bool = True) -> LoadPlan:
    """
    Create the ``LoadPlan`` used by DefaultLoadPlanner.

    This function generates a local load plan for the given state dictionary and metadata.
    It iterates over each key-value pair in the state dictionary, and for each key, it tries to
    find the corresponding metadata. If the key is missing in the metadata and the `strict` flag
    is set to `True`, it raises an exception. Otherwise, it logs a warning and continues.

    For each valid key, it creates read items based on the object type. If the object is a DTensor,
    it checks if the current rank is part of the mesh for the corresponding DTensor before creating
    read items.

    Args:
        state_dict (Dict[str, Any]): A dictionary representing the state to be loaded.
        metadata (Metadata): Metadata associated with the state dictionary.
        strict (bool, optional): If True, raise an exception for missing keys in the metadata.
                                 Defaults to True.

    Returns:
        LoadPlan: A load plan containing read requests for the state dictionary.
    """
    requests = []
    for fqn, obj in state_dict.items():
        try:
            md = metadata.state_dict_metadata[fqn]
        except Exception as e:
            if strict:
                raise e
            else:
                logger.warning("Ignore missing key %s when loading checkpoint as strict=%s", fqn, strict)
                continue
        logger.debug("Creating load plan for %s successfully", fqn)
        # Since DTensor supports submesh, adding extra check to ensure _create_read_items()
        # gets called only when the current rank is part of the mesh for the corresponding DTensor.
        if isinstance(obj, DTensor):
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_read_items(fqn, md, obj)
        else:
            requests += _create_read_items(fqn, md, obj)

    return LoadPlan(requests)


def create_default_global_load_plan(all_plans: List[LoadPlan]) -> List[LoadPlan]:
    """
    Create global load plan used by DefaultLoadPlanner.

    The default load behavior involved no global coordination and this function
    currently doesn't change the local plans.

    Args:
        all_plans (List[LoadPlan]): A list of local load plans.

    Returns:
        List[LoadPlan]: The same list of local load plans without any modification.
    """
    return all_plans


def create_default_local_save_plan(state_dict: Dict[str, Any], is_coordinator: bool) -> SavePlan:
    """
    Create the ``SavePlan`` used by DefaultSavePlanner.

    On non-coordinator ranks, this function ignores tensors and non-tensor objects,
    only producing writes for ShardedTensor objects.

    On the coordinator rank, produce writes for all values.

    Args:
        state_dict (Dict[str, Any]): A dictionary representing the state to be saved.
        is_coordinator (bool): A flag indicating whether the current process is the coordinator.

    Returns:
        SavePlan: A save plan containing write requests for the state dictionary.
    """
    requests = []
    for fqn, obj in state_dict.items():
        # Since DTensor supports submesh, adding extra check to ensure _create_write_items()
        # gets called only when the current rank is part of the mesh for the corresponding DTensor.
        if isinstance(obj, DTensor):
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_write_items(fqn, obj)
        elif isinstance(obj, (torch.Tensor)) or is_coordinator:
            requests += _create_write_items(fqn, obj)

    return SavePlan(requests)


def create_default_global_save_plan(
    all_plans: List[SavePlan],
    rewrite_index_hints: bool = True,
) -> Tuple[List[SavePlan], Metadata]:
    """
    Create the global plan and metadata used by DefaultSavePlanner.

    Metadata is produced by concatenating the metadata of all ``WriteItem`` from the supplied plans.

    The only global planning change is to update index hints in all ``MetadataIndex`` objects if
    ``rewrite_index_hints`` is True.

    Args:
        all_plans (List[SavePlan]): A list of local save plans.
        rewrite_index_hints (bool, optional): A flag indicating whether to rewrite index hints. Defaults to True.

    Returns:
        Tuple[List[SavePlan], Metadata]: A tuple containing the global save plan and the associated metadata.
    """
    md: Dict[str, STORAGE_TYPES] = {}
    new_plans = []
    for plan in all_plans:
        new_items = []
        for item in plan.items:
            if not item.type == WriteItemType.SHARD:
                assert item.index.fqn not in md

            if item.type == WriteItemType.BYTE_IO:
                md[item.index.fqn] = BytesStorageMetadata()
                new_items.append(item)
            else:
                assert item.tensor_data is not None
                tensor_md = cast(
                    TensorStorageMetadata,
                    md.setdefault(
                        item.index.fqn,
                        TensorStorageMetadata(
                            properties=item.tensor_data.properties,
                            size=item.tensor_data.size,
                            chunks=[],
                        ),
                    ),
                )
                new_item = item
                if rewrite_index_hints:
                    new_index = dataclasses.replace(item.index, index=len(tensor_md.chunks))
                    new_item = dataclasses.replace(item, index=new_index)
                new_items.append(new_item)

                assert item.tensor_data.chunk is not None, f"""
                    Cannot create MD for tensor without bounds.
                    FQN: {item.index.fqn}
                """
                tensor_md.chunks.append(item.tensor_data.chunk)
        new_plans.append(dataclasses.replace(plan, items=new_items))
    return (new_plans, Metadata(md))


def _create_default_local_metadata(state_dict: STATE_DICT_TYPE) -> Metadata:
    """
    Return the ``Metadata`` if DefaultSavePlanner was used to checkpoint ``state_dict``.
    """
    plan = _create_default_metadata_only_plan(state_dict)
    _, md = create_default_global_save_plan([plan])
    return md


def _check_box_overlap(box0: ChunkStorageMetadata, box1: ChunkStorageMetadata) -> bool:
    """
    Checks if two boxes represented by `ChunkStorageMetadata` objects overlap.

    Args:
        box0 (ChunkStorageMetadata): The first box to check for overlap.
        box1 (ChunkStorageMetadata): The second box to check for overlap.

    Returns:
        bool: True if the boxes overlap, False otherwise.

    Notes:
        Each box is defined by its offset and size in each dimension.
        The function iterates through each dimension to determine if the boxes overlap.
        If, in any dimension, one box is completely to the left or above the other,
        then the boxes do not overlap.
    """

    # For each dim of each shard, check if one shard resides on the other
    # end of second shard with respect to that dim. As an example for a 2D
    # shard, we would check if one shard is above or on the left of the
    # other shard.
    ndims = len(box0.offsets)
    for i in range(ndims):
        if box0.offsets[i] >= box1.offsets[i] + box1.sizes[i]:
            return False
        if box1.offsets[i] >= box0.offsets[i] + box0.sizes[i]:
            return False

    return True


def _check_box_bounds(outer_box_size: torch.Size, inner_box: ChunkStorageMetadata) -> bool:
    """
    Checks if the inner box is entirely within the bounds of the outer box.

    This function iterates through each dimension of the outer box and the inner box.
    It verifies that the inner box's offset is non-negative, its size is non-negative,
    and the sum of its offset and size does not exceed the outer box's size in each dimension.

    Args:
        outer_box_size (torch.Size): The size of the outer box in each dimension.
        inner_box (ChunkStorageMetadata): The metadata of the inner box, including its offsets and sizes.

    Returns:
        bool: True if the inner box is entirely within the bounds of the outer box, False otherwise.
    """
    for i in range(len(outer_box_size)):
        if inner_box.offsets[i] < 0:
            return False
        if inner_box.sizes[i] < 0:
            return False
        if inner_box.offsets[i] + inner_box.sizes[i] > outer_box_size[i]:
            return False

    return True


def _validate_global_plan(global_plan: List[SavePlan], metadata: Metadata) -> bool:
    """
    Validate the global save plan and associated metadata.

    This function checks the integrity of the metadata for each key in the state dictionary.
    It verifies that each tensor's chunks are within the tensor's bounds, do not overlap,
    and together cover the entire tensor.

    Args:
        global_plan (List[SavePlan]): A list of global save plans.
        metadata (Metadata): The metadata associated with the state dictionary.

    Returns:
        bool: True if all checks pass, False otherwise.
    """
    all_good = True
    for key, value in metadata.state_dict_metadata.items():
        if isinstance(value, BytesStorageMetadata):
            continue
        if len(value.size) == 0:
            continue
        chunks_volume = 0
        for chunk_idx, chunk0 in enumerate(value.chunks):
            # Compute the volume.
            if not _check_box_bounds(value.size, chunk0):
                logger.warning(
                    """
                        key:%s has out of bounds chunk:
                        tensor-size:%s chunk: %s
                    """,
                    key,
                    value.size,
                    chunk0,
                )
                all_good = False
            chunks_volume += reduce(operator.mul, chunk0.sizes, 1)

            # Check for overlap
            for chunk1 in value.chunks[chunk_idx + 1 :]:
                if _check_box_overlap(chunk0, chunk1):
                    logger.warning("key:%s has overlapping chunks: %s %s", key, chunk0, chunk1)
                    all_good = False

        # Check whether combined chunk cover the whole tensor.
        tensor_volume = reduce(operator.mul, value.size, 1)
        if chunks_volume != tensor_volume:
            logger.warning(
                """
                    key:%s invalid fill tensor-volume:
                    %s chunks-volume: %s
                """,
                key,
                tensor_volume,
                chunks_volume,
            )
            all_good = False

    return all_good
