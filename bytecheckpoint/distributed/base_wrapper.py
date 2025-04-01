################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
################################################################################

from itertools import chain
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast

import torch.distributed as dist
from torch.distributed.checkpoint.api import (
    WRAPPED_EXCEPTION,
    CheckpointException,
    _wrap_exception,
)
from torch.distributed.checkpoint.utils import _DistWrapper, _get_failure_dict

T = TypeVar("T")
R = TypeVar("R")

# TODO: Support more parallelism strategies.
INTRA_NODE_DP_SIZE = 8
DP_LEVEL = 2
TP_DP_PP_LEVEL = 3

LEVEL_TREE_NAMES = ["DP_LEVEL", "TP_DP_PP_LEVEL"]

LEVEL_TREE_CACHE: Dict[str, Optional[List[Optional[Tuple[int, dist.ProcessGroup]]]]] = {
    "DP_LEVEL": None,
    "TP_DP_PP_LEVEL": None,
}


class _BaseWrapper(_DistWrapper):
    def __init__(self, group: Optional[dist.ProcessGroup], use_dist: bool, coordinator_rank: int):
        super().__init__(group, use_dist, coordinator_rank)

    def broadcast_as_scatter_object(self, object: List[T]) -> T:
        """
        Broadcast an object list and then scatter the results to each rank.

        This method first broadcasts the given object list to all ranks in the process group.
        Then, it divides the broadcasted results into equal partitions based on the group size
        and retrieves the partition corresponding to the current rank.

        Args:
            object (List[T]): The object list to be broadcasted.

        Returns:
            T: The partition of the broadcasted results corresponding to the current rank.
        """
        broadcast_results = self.broadcast_object(object)
        # Fetch local results from the broadcasted results.
        group_size = dist.get_world_size(self.group)
        assert len(broadcast_results) % group_size == 0
        partition_size = len(broadcast_results) // group_size
        my_group_rank = dist.get_rank(self.group)
        all_results = broadcast_results[my_group_rank * partition_size : (my_group_rank + 1) * partition_size]
        return all_results

    def reduce_scatter(
        self,
        step: str,
        map_fun: Callable[[], T],
        reduce_fun: Callable[[List[T]], List[R]],
    ) -> R:
        """
        Cover the implementation of the torch distributed module for stable communication primitives.
        This method computes a value on each rank, then performs a centralized reduction on a single rank,
        followed by scattering the result to each rank.

        Args:
            step (str): A string representing the current step or stage of the operation.
            map_fun (Callable[[], T]): A function that computes a value on each rank.
            reduce_fun (Callable[[List[T]], List[R]]): A function that performs a reduction on the gathered values.

        Returns:
            R: The result of the reduction and scatter operation.

        Raises:
            CheckpointException: If any rank raises an exception during the operation.

        This method operates in the following way:
            1. Run `map_fun` on all ranks.
            2. Gather results on rank 0.
            3. Call `reduce_fun` on all those values.
            4. Scatter the reduced result to each rank.
        """
        local_data: Union[WRAPPED_EXCEPTION, T]
        try:
            local_data = map_fun()
        except BaseException as e:
            local_data = _wrap_exception(e)

        # N.B. we use all_gather_object() instead of hier_gather_object() for better communication stability.
        all_data = self.all_gather_object(local_data)

        all_results: Optional[List[Union[R, CheckpointException]]] = None
        if self.is_coordinator:
            assert all_data is not None
            node_failures = _get_failure_dict(all_data)

            if len(node_failures) == 0:
                try:
                    # N.B. why can't mypy cast List[R] to List[Union[R, WRAPPED_EXCEPTION]]?
                    all_results = cast(
                        List[Union[R, CheckpointException]],
                        reduce_fun(cast(List[T], all_data)),
                    )
                except BaseException as e:
                    node_failures[self.rank] = _wrap_exception(e)

            if len(node_failures) > 0:
                all_results = [CheckpointException(step, node_failures)] * self.get_world_size()

        # N.B. we use broadcast_object() to scatter results for better communication stability.
        result = self.broadcast_as_scatter_object(all_results)
        # result = self.scatter_object(all_results)

        assert len(result) == 1
        result = result[0]
        if isinstance(result, CheckpointException):
            raise result
        return result

    def all_reduce(
        self,
        step: str,
        map_fun: Callable[[], T],
        reduce_fun: Callable[[List[T]], List[R]],
    ) -> R:
        """
        Cover the implementation of the torch distributed module for stable communication primitives.
        This method computes a value on each rank, then performs a centralized reduction on a single rank,
        followed by broadcasting the reduced value to all ranks.

        Args:
            step (str): A string representing the current step or stage of the operation.
            map_fun (Callable[[], T]): A function that computes a value on each rank.
            reduce_fun (Callable[[List[T]], List[R]]): A function that performs a reduction on the gathered values.

        Returns:
            R: The result of the reduction and broadcast operation.

        Raises:
            CheckpointException: If any rank raises an exception during the operation.

        This method operates in the following way:
            1. Run `map_fun` on all ranks.
            2. Gather results on rank 0.
            3. Call `reduce_fun` on all those values.
            4. Broadcast the reduced value to all ranks.
        """
        local_data: Union[T, WRAPPED_EXCEPTION]
        try:
            local_data = map_fun()
        except BaseException as e:
            local_data = _wrap_exception(e)

        all_data = self.all_gather_object(local_data)

        result: Optional[Union[R, CheckpointException]] = None
        if self.is_coordinator:
            assert all_data is not None
            node_failures = _get_failure_dict(all_data)
            if len(node_failures) == 0:
                try:
                    result = reduce_fun(cast(List[T], all_data))
                except BaseException as e:
                    node_failures[self.rank] = _wrap_exception(e)

            if len(node_failures) > 0:
                result = CheckpointException(step, node_failures)

        final_result = self.broadcast_object(result)
        if isinstance(final_result, CheckpointException):
            raise final_result
        return cast(R, final_result)


class _BaseTreeTopoWrapper(_BaseWrapper):
    """
    This is a wrapper inherited from `_DistWrapper` around PG that provides a series

    of features around object hierarchical collectives.

    It works without distributed initialized, where most collectives turns into nops.

    All variants that take functions are exception robust, meaning that if one or more
    ranks raise errors, all ranks will observe those.
    """

    def __init__(
        self,
        group: Optional[dist.ProcessGroup],
        use_dist: bool,
        coordinator_rank: int,
        enable_tree_topo: bool = False,
    ):
        super().__init__(group, use_dist, coordinator_rank)
        self.is_constructed = False
        self.level: int = None
        self.level_trees = None
        if enable_tree_topo:
            self.construct_topo()

    def construct_topo(self):
        """
        Protocol to construct the tree-based hierarchical communication topology.

        Implementations should correctly set the values of self.level_trees (root_rank, prcocess_group)
        according to the specific training frameworks and the parallelism strategies.

        NOTE: currently, only 1D (FSDP), parallelism are supported.
        """
        raise NotImplementedError

    def cache_level_trees(self, tree_name: str):
        assert tree_name in LEVEL_TREE_NAMES
        LEVEL_TREE_CACHE[tree_name] = self.level_trees

    def _construct_topo_binary(self):
        """
        Protocol to construct the tree-based hierarchical communication topology.

        TODO: The middle and top level forests obey the binary tree construction rules.

        Implementations should correctly set the values of self.level_roots and self.level_groups
        according to the specific training frameworks and the parallelism strategies.

        """
        raise NotImplementedError

    def _check_reduce_funs(self, reduce_funs: List[Optional[Callable[[List[T]], List[R]]]]) -> None:
        assert len(self.level_trees) == len(reduce_funs), (
            f"The number of reduce_funs: {len(reduce_funs)} does not match the number of level trees: {len(self.level_trees)}"
        )

    def get_root_ranks(self) -> List[Optional[int]]:
        """
        Retrieve the root ranks from the level trees.

        This method iterates over the `self.level_trees` list, which contains tuples
        representing node data. For each tuple, it extracts the root rank (the first
        element of the tuple) if the node data is not None. If the node data is None,
        it appends None to the list of root ranks.

        Returns:
            List[Optional[int]]: A list containing the root ranks from the level trees.
                                 If a node data is None, the corresponding element in
                                 the list will be None.
        """
        root_ranks = []
        for node_data in self.level_trees:
            if node_data is not None:
                root_ranks.append(node_data[0])
            else:
                root_ranks.append(None)
        return root_ranks

    def hier_gather_object(
        self,
        objects: List[T],
        dst: int = 0,
        group: Optional[dist.ProcessGroup] = None,
    ) -> Optional[List[T]]:
        """
        Implement functionality similar to c10d::gather_object.

        This method gathers a list of objects from all ranks in the process group to a destination rank.
        If distributed communication is enabled (`self.use_dist` is True), it uses `dist.gather_object` to perform the gathering.
        Otherwise, it simply returns the input objects.

        Args:
            objects (List[T]): The list of objects to be gathered.
            dst (int, optional): The destination rank where the objects will be gathered. Defaults to 0.
            group (Optional[dist.ProcessGroup], optional): The process group to perform the gathering within. Defaults to None.

        Returns:
            Optional[List[T]]: A list containing all the gathered objects if the current rank is the destination rank,
                               or the input objects if distributed communication is disabled. Otherwise, returns None.
        """
        if self.use_dist:
            if dist.get_rank() == dst:
                gather_objs = cast(List[List[T]], [None] * dist.get_world_size(group))
            else:
                gather_objs = None
            dist.gather_object(
                obj=objects,
                object_gather_list=gather_objs,
                dst=dst,
                group=group,
            )
            result = gather_objs
            if result is not None:
                result: List[T] = list(chain.from_iterable(result))
        else:
            result = objects
        return result

    def hier_scatter_object(
        self,
        object_list: Optional[List[T]],
        src: int = 0,
        group: Optional[dist.ProcessGroup] = None,
    ) -> List[T]:
        """
        Implement functionality similar to c10d::scatter_object.

        This method scatters a list of objects from a source rank to all ranks in the process group.
        If distributed communication is enabled (`self.use_dist` is True), it uses `dist.scatter_object_list` to perform the scattering.
        Otherwise, it simply returns the first element of the input object list.

        Args:
            object_list (Optional[List[T]]): The list of objects to be scattered.
            src (int, optional): The source rank where the objects will be scattered from. Defaults to 0.
            group (Optional[dist.ProcessGroup], optional): The process group to perform the scattering within. Defaults to None.

        Returns:
            List[T]: The partition of the scattered objects corresponding to the current rank.
        """
        if self.use_dist:
            gather_result = cast(List[List[T]], [None])
            if dist.get_rank() == src:
                group_size = dist.get_world_size(group)
                assert len(object_list) % group_size == 0
                partition_size = len(object_list) // group_size
                scatter_objs: List[List[T]] = [
                    object_list[idx : idx + partition_size] for idx in range(0, len(object_list), partition_size)
                ]
            else:
                scatter_objs = None
            dist.scatter_object_list(
                scatter_object_output_list=gather_result,
                scatter_object_input_list=scatter_objs,
                src=src,
                group=group,
            )

            local_reply = gather_result[0]
        else:
            assert object_list is not None
            local_reply = [object_list[0]]
        return local_reply

    def hier_broadcast_object(self, object: Optional[T], src: int = 0, group: Optional[dist.ProcessGroup] = None) -> T:
        """
        Implement functionality similar to c10d::broad_object_list.

        This method broadcasts an optional object from a source rank to all ranks in the process group.
        If distributed communication is enabled (`self.use_dist` is True), it uses `dist.broadcast_object_list`
        to perform the broadcast. Otherwise, it simply returns the input object.

        Args:
            object (Optional[T]): The object to be broadcasted.
            src (int, optional): The source rank where the object will be broadcasted from. Defaults to 0.
            group (Optional[dist.ProcessGroup], optional): The process group to perform the broadcast within. Defaults to None.

        Returns:
            T: The broadcasted object.
        """
        object_list = [object]
        if self.use_dist:
            dist.broadcast_object_list(object_list=object_list, src=src, group=group)
        return cast(T, object_list[0])

    def hier_broadcast_as_scatter_object(
        self, object: Optional[T], src: int = 0, group: Optional[dist.ProcessGroup] = None
    ) -> T:
        """
        Broadcast an object and then scatter the results to each rank.

        This method first broadcasts the given object to all ranks in the process group.
        Then, it divides the broadcasted results into equal partitions based on the group size
        and retrieves the partition corresponding to the current rank.

        Args:
            object (Optional[T]): The object to be broadcasted.
            src (int, optional): The source rank where the object will be broadcasted from. Defaults to 0.
            group (Optional[dist.ProcessGroup], optional): The process group to perform the broadcast and scatter within. Defaults to None.

        Returns:
            T: The partition of the broadcasted results corresponding to the current rank.

        Raises:
            AssertionError: If the length of the broadcasted results is not divisible by the group size.
        """
        broadcast_results = self.hier_broadcast_object(object, src, group)
        # Fetch local results from the broadcasted results.
        group_size = dist.get_world_size(group)
        assert len(broadcast_results) % group_size == 0
        partition_size = len(broadcast_results) // group_size
        my_group_rank = dist.get_rank(group)
        all_results = broadcast_results[my_group_rank * partition_size : (my_group_rank + 1) * partition_size]
        return all_results

    def hier_all_gather_object(self, objects: List[T], group: Optional[dist.ProcessGroup] = None) -> List[T]:
        """
        Implement functionality similar to c10d::all_gather_object.

        This method gathers a list of objects from all ranks in the process group.
        If distributed communication is enabled (`self.use_dist` is True), it uses `dist.all_gather_object` to perform the gathering.
        Otherwise, it simply returns the input objects.

        Args:
            objects (List[T]): The list of objects to be gathered.
            group (Optional[dist.ProcessGroup], optional): The process group to perform the gathering within. Defaults to None.

        Returns:
            List[T]: A list containing all the gathered objects.
        """
        if self.use_dist:
            gather_objs = cast(List[List[T]], [None] * dist.get_world_size(group))
            dist.all_gather_object(object_list=gather_objs, obj=objects, group=group)
            result = gather_objs
            if result is not None:
                result: List[T] = list(chain.from_iterable(result))
        else:
            result = objects
        return result

    def hier_reduce_scatter(
        self,
        step: str,
        map_fun: Callable[[], T],
        reduce_funs: List[Optional[Callable[[List[T]], List[R]]]],
    ) -> R:
        """
        Perform a hierarchical reduce-scatter operation across a communication tree.

        This method first computes a local value on each rank using `map_fun`, then gathers these values
        up the communication tree, applies reduction functions at each level, and finally scatters the
        reduced results back down the tree to each rank.

        Args:
            step (str): A string representing the current step or stage of the operation.
            map_fun (Callable[[], T]): A function that computes a local value on each rank.
            reduce_funs (List[Optional[Callable[[List[T]], List[R]]]]): A list of reduction functions,
                one for each level of the communication tree. If a level does not require a reduction,
                the corresponding function can be `None`.

        Returns:
            R: The result of the reduce-scatter operation for the current rank.

        Raises:
            AssertionError: If the communication tree is not constructed, or if a root rank does not have all data.
            CheckpointException: If any rank raises an exception during the operation.
        """
        self._check_reduce_funs(reduce_funs)
        assert self.is_constructed
        # Step 1: call map_fun() on all ranks.
        local_data: Union[WRAPPED_EXCEPTION, T]
        try:
            local_data = [map_fun()]
        except BaseException as e:
            local_data = [_wrap_exception(e)]

        # Step 2: call hier_gather_object() on the communication tree.
        all_data = local_data
        for node_data, reduce_fun in zip(self.level_trees, reduce_funs):
            if node_data is not None:
                root_rank, group = node_data

                # N.B. we use all_gather_object() instead of hier_gather_object() for better communication stability.
                all_data = self.hier_all_gather_object(all_data, group)
                if dist.get_rank() == root_rank:
                    assert all_data is not None, f"root rank: {root_rank}, does not have all data"
                    if reduce_fun:
                        all_data = cast(List[Union[R, CheckpointException]], reduce_fun(cast(List[T], all_data)))
                else:
                    all_data = None

        # Step 3: call hier_scatter_object() on the communication tree.
        all_results: Optional[List[Union[R, CheckpointException]]] = None

        # Scatter from the top-level coordinator.
        root_ranks = self.get_root_ranks()
        if root_ranks[-1] is not None and dist.get_rank() == root_ranks[-1]:
            assert all_data is not None, (
                f"coordinator {self.coordinator_rank} does not have all data, my global rank: {dist.get_rank()}"
            )
            all_results = all_data
            node_failures = _get_failure_dict(all_data)

            if len(node_failures) > 0:
                all_results = [CheckpointException(step, node_failures)] * self.get_world_size()

        for node_data in self.level_trees[::-1]:
            if node_data is not None:
                root_rank, group = node_data

                all_results = self.hier_broadcast_as_scatter_object(all_results, root_rank, group)

        # Get final result.
        assert len(all_results) == 1
        result = all_results[0]
        if isinstance(result, CheckpointException):
            raise result
        return result

    def hier_all_reduce(
        self,
        step: str,
        map_fun: Callable[[], T],
        reduce_funs: List[Optional[Callable[[List[T]], R]]],
    ) -> R:
        """
        Perform a hierarchical all-reduce operation across a communication tree.

        This method first computes a local value on each rank using `map_fun`, then gathers these values
        up the communication tree, applies reduction functions at each level, and finally broadcasts the
        reduced result back down the tree to all ranks.

        Args:
            step (str): A string representing the current step or stage of the operation.
            map_fun (Callable[[], T]): A function that computes a local value on each rank.
            reduce_funs (List[Optional[Callable[[List[T]], R]]]): A list of reduction functions,
                one for each level of the communication tree. If a level does not require a reduction,
                the corresponding function can be `None`.

        Returns:
            R: The result of the all-reduce operation for the current rank.

        Raises:
            AssertionError: If the communication tree is not constructed, or if a root rank does not have all data.
            CheckpointException: If any rank raises an exception during the operation.
        """
        self._check_reduce_funs(reduce_funs)
        assert self.is_constructed
        # Step 1: call map_fun() on all ranks.
        local_data: Union[WRAPPED_EXCEPTION, T]
        try:
            local_data = [map_fun()]
        except BaseException as e:
            local_data = [_wrap_exception(e)]

        # Step 2: call hier_gather_object() on the communication tree.
        all_data = local_data
        for node_data, reduce_fun in zip(self.level_trees, reduce_funs):
            if node_data is not None:
                root_rank, group = node_data
                # N.B. we use all_gather_object() instead of hier_gather_object() for better communication stability.
                all_data = self.hier_all_gather_object(all_data, group)
                if dist.get_rank() == root_rank:
                    assert all_data is not None, f"root rank: {root_rank}, does not have all data"
                    if reduce_fun:
                        all_data = [cast(Union[R, CheckpointException], reduce_fun(cast(List[T], all_data)))]
                else:
                    all_data = None

        # Step 3: call hier_broadcast_object() on the communication tree.
        result: Optional[Union[R, CheckpointException]] = None

        # broadcast from the top-level coordinator.
        root_ranks = self.get_root_ranks()
        if root_ranks[-1] is not None and dist.get_rank() == root_ranks[-1]:
            assert all_data is not None
            assert len(all_data) == 1
            node_failures = _get_failure_dict(all_data)
            result = all_data[0]

            if len(node_failures) > 0:
                result = CheckpointException(step, node_failures)

        for node_data in self.level_trees[::-1]:
            if node_data is not None:
                root_rank, group = node_data
                result = self.hier_broadcast_object(result, root_rank, group)

        # Get final result.
        if isinstance(result, CheckpointException):
            raise result
        return cast(R, result)

    def exec_reduce_scatter(
        self,
        step: str,
        map_fun: Callable[[], T],
        reduce_fun: Callable[[List[T]], List[R]],
    ) -> R:
        """
        Perform a reduce-scatter operation, either hierarchically or non-hierarchically.

        This method decides whether to perform a hierarchical or non-hierarchical reduce-scatter operation
        based on the construction status of the communication tree. If the tree is constructed, it calls
        `hier_reduce_scatter` with appropriate reduction functions. Otherwise, it calls `reduce_scatter`.

        Args:
            step (str): A string representing the current step or stage of the operation.
            map_fun (Callable[[], T]): A function that computes a local value on each rank.
            reduce_fun (Callable[[List[T]], List[R]]): A reduction function to be applied to the gathered values.

        Returns:
            R: The result of the reduce-scatter operation for the current rank.

        Raises:
            AssertionError: If the communication tree is constructed but the level is not valid.
        """
        if self.is_constructed:
            assert self.level is not None and self.level > 1
            reduce_funs = [None] * (self.level - 1)
            reduce_funs.append(reduce_fun)
            return self.hier_reduce_scatter(step, map_fun, reduce_funs)
        else:
            return self.reduce_scatter(step, map_fun, reduce_fun)

    def exec_all_reduce(
        self,
        step: str,
        map_fun: Callable[[], T],
        reduce_fun: Optional[Callable[[List[T]], R]],
    ) -> R:
        """
        Perform an all-reduce operation, either hierarchically or non-hierarchically.

        This method decides whether to perform a hierarchical or non-hierarchical all-reduce operation
        based on the construction status of the communication tree. If the tree is constructed, it calls
        `hier_all_reduce` with appropriate reduction functions. Otherwise, it calls `all_reduce`.

        Args:
            step (str): A string representing the current step or stage of the operation.
            map_fun (Callable[[], T]): A function that computes a local value on each rank.
            reduce_fun (Optional[Callable[[List[T]], R]]): A reduction function to be applied to the gathered values.

        Returns:
            R: The result of the all-reduce operation for the current rank.

        Raises:
            AssertionError: If the communication tree is constructed but the level is not valid.
        """
        if self.is_constructed:
            assert self.level is not None and self.level > 1
            reduce_funs = [None] * (self.level - 1)
            reduce_funs.append(reduce_fun)
            return self.hier_all_reduce(step, map_fun, reduce_funs)
        else:
            return self.all_reduce(step, map_fun, reduce_fun)
