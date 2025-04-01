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
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, List, Optional, Set, Union

import torch
import torch.distributed as dist

from bytecheckpoint.io import bfile
from bytecheckpoint.planner.default_planner import DefaultLoadPlanner, DefaultSavePlanner
from bytecheckpoint.storage import CKPTCounter
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger
from bytecheckpoint.workflow.extra_state import load_extra_state, save_extra_state
from bytecheckpoint.workflow.state_dict.load_state_dict import load_state_dict
from bytecheckpoint.workflow.state_dict.save_state_dict import save_state_dict

from .meta_type import (
    _DIRECTORY_FORMAT,
    EXTRA_STATE_STR,
    MODEL_STR,
    OPTIMIZER_STR,
    STATE_DICT_TYPE,
    CheckpointState,
)

SUPPORTED_TYPES = {MODEL_STR, OPTIMIZER_STR, EXTRA_STATE_STR}


logger = get_bytecheckpoint_logger()


class _BaseCheckpointer(ABC):
    """
    The Checkpointer class offers APIs that enable users to save and load state dictionarie.
    It is designed for extension across various training frameworks.
    """

    @classmethod
    @abstractmethod
    def save(cls, ckpt_path: str, checkpoint_state: CheckpointState, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def _save_model(cls, model_path: str, model_state: STATE_DICT_TYPE, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def _save_optimizer(cls, optimizer_path: str, optimizer_state: STATE_DICT_TYPE, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def _save_extra_state(cls, extra_state_path: str, extra_state: STATE_DICT_TYPE, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def load(cls, ckpt_path: str, checkpoint_state: CheckpointState, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def _load_model(cls, model_path: str, model_state: STATE_DICT_TYPE, *args, **kwargs) -> STATE_DICT_TYPE:
        pass

    @classmethod
    @abstractmethod
    def _load_optimizer(cls, optimizer_path: str, optimizer_state: STATE_DICT_TYPE, *args, **kwargs) -> STATE_DICT_TYPE:
        pass

    @classmethod
    @abstractmethod
    def _load_extra_state(cls, extra_state_path: str, extra_state: STATE_DICT_TYPE, *args, **kwargs) -> STATE_DICT_TYPE:
        pass


class BaseCheckpointer(_BaseCheckpointer):
    """
    Basic implementation of the Checkpointer class.
    """

    _framework_name = None

    @classmethod
    def save(
        cls,
        ckpt_path: str,
        checkpoint_state: CheckpointState,
        fast_saving: bool = False,
        global_steps: Optional[int] = None,
    ):
        """
        A Method for saving checkpoint

        Args:
            ckpt_path (str): Root directory where the checkpoint will be saved.
            checkpoint_state (CheckpointState): A dictionary contains key-value pairs for model and optimizer.
            fast_saving (bool, optional): A boolean value indicating if saving checkpoint asynchronously,
                i.e. after dumping tensors from GPU memory to Host memory (D2H), the training program can continue training. Defaults to False.
            global_steps (Optional[int], optional): The global step number at the time of saving the checkpoint. Defaults to None.

        Returns:
            None

        Raises:
            NotImplementedError: This method is not implemented and should be overridden by subclasses.
        """
        raise NotImplementedError

    @classmethod
    def load(
        cls,
        ckpt_path: str,
        checkpoint_state: CheckpointState,
        fast_loading: bool = False,
        strict: bool = True,
    ):
        """
        A Method for loading checkpoint

        Args:
            ckpt_path (str): Root directory where the checkpoint is stored.
            checkpoint_state (CheckpointState): A dictionary contains key-value pairs for model and optimizer.
            fast_loading (bool, optional): A boolean value indicating if loading checkpoint in parallel. Defaults to False.
            strict (bool, optional): Whether to strictly enforce that all keys in the checkpoint match the model/optimizer state dictionary. Defaults to True.

        Returns:
            None

        Raises:
            NotImplementedError: This method is not implemented and should be overridden by subclasses.
        """
        raise NotImplementedError

    """
    Checkpint saving methods.
    """

    @classmethod
    def _save_model(
        cls,
        model_path: str,
        model_state: STATE_DICT_TYPE,
        framework_name: str,
        role: str,
        save_planner: DefaultSavePlanner,
        ckpt_path: str,
        save_ckpt_start_time: float,
        ckpt_counter: CKPTCounter,
        global_steps: Optional[int],
        fast_saving: bool,
        callback: Optional[Callable],
    ):
        # Save model.
        save_state_dict(
            state_dict=model_state,
            state_path=model_path,
            root_path=ckpt_path,
            ckpt_name=MODEL_STR,
            framework_name=framework_name,
            suffix=role if role else "",
            planner=save_planner,
            save_ckpt_start_time=save_ckpt_start_time,
            ckpt_counter=ckpt_counter,
            global_steps=global_steps,
            callback=callback,
            pg_or_mesh=None,
            coordinator_rank=0,
            no_dist=False,
            async_io=fast_saving,
        )

    @classmethod
    def _save_optimizer(
        cls,
        optimizer_path: str,
        optimizer_state: STATE_DICT_TYPE,
        framework_name: str,
        role: str,
        save_planner: DefaultSavePlanner,
        ckpt_path,
        save_ckpt_start_time,
        ckpt_counter: CKPTCounter,
        global_steps: int,
        fast_saving: bool,
        callback: Optional[Callable],
    ):
        # Save optimizer.
        save_state_dict(
            state_dict=optimizer_state,
            state_path=optimizer_path,
            root_path=ckpt_path,
            ckpt_name=OPTIMIZER_STR,
            framework_name=framework_name,
            suffix=role if role else "",
            planner=save_planner,
            save_ckpt_start_time=save_ckpt_start_time,
            ckpt_counter=ckpt_counter,
            global_steps=global_steps,
            callback=callback,
            pg_or_mesh=None,
            coordinator_rank=0,
            no_dist=False,
            async_io=fast_saving,
        )

    @classmethod
    def _save_extra_state(
        cls,
        extra_state_path: str,
        extra_state: STATE_DICT_TYPE,
        extra_state_file_name: STATE_DICT_TYPE,
        framework_name: str,
        role: str,
        ckpt_path,
        save_ckpt_start_time,
        ckpt_counter: CKPTCounter,
        global_steps: int,
        fast_saving: bool,
        callback: Optional[Callable],
    ):
        # Save extra state.
        save_extra_state(
            file_name=extra_state_file_name,
            state_path=extra_state_path,
            root_path=ckpt_path,
            state_dict=extra_state,
            ckpt_name=EXTRA_STATE_STR,
            framework_name=framework_name,
            suffix=role if role else "",
            save_ckpt_start_time=save_ckpt_start_time,
            ckpt_counter=ckpt_counter,
            global_steps=global_steps,
            callback=callback,
            async_io=fast_saving,
        )

    """
    Checkpint loading methods.
    """

    @classmethod
    def _load_model(
        cls,
        model_path: str,
        model_state: STATE_DICT_TYPE,
        framework_name: str,
        role: str,
        load_planner: DefaultLoadPlanner,
        fast_loading: bool,
    ) -> STATE_DICT_TYPE:
        # Load model.
        model_load_futures = load_state_dict(
            state_dict=model_state,
            path=model_path,
            ckpt_name=MODEL_STR,
            framework_name=framework_name,
            suffix=role if role else "",
            planner=load_planner,
            planning_pg_or_mesh=None,
            coordinator_rank=0,
            no_dist=False,
            fast_loading=fast_loading,
        )
        # Process model state dict.
        for future in model_load_futures:
            future.result()
        return model_state

    @classmethod
    def _load_optimizer(
        cls,
        optimizer_path: str,
        optimizer_state: STATE_DICT_TYPE,
        framework_name: str,
        role: str,
        load_planner: DefaultLoadPlanner,
        fast_loading: bool,
    ) -> STATE_DICT_TYPE:
        # Load optimizer.
        optimizer_load_futures = load_state_dict(
            state_dict=optimizer_state,
            path=optimizer_path,
            ckpt_name=OPTIMIZER_STR,
            framework_name=framework_name,
            suffix=role if role else "",
            planner=load_planner,
            planning_pg_or_mesh=None,
            coordinator_rank=0,
            no_dist=False,
            fast_loading=fast_loading,
        )
        # Process optimizer state dict.
        for future in optimizer_load_futures:
            future.result()
        return optimizer_state

    @classmethod
    def _load_extra_state(
        cls,
        extra_state_path: str,
        extra_state_file_name: str,
        framework_name: str,
        role: str,
    ) -> STATE_DICT_TYPE:
        extra_state_load_futures = load_extra_state(
            file_name=extra_state_file_name,
            path=extra_state_path,
            ckpt_name=EXTRA_STATE_STR,
            framework_name=framework_name,
            suffix=role if role else "",
        )
        return extra_state_load_futures[0].result()

    """
    Helper methods.
    """

    @classmethod
    def get_curr_ckpt_path(
        cls,
        ckpt_path: str,
        global_steps: Optional[int],
        ignore_append_global_steps_to_folder: Optional[bool],
    ):
        curr_ckpt_path = ckpt_path
        if global_steps is not None and not ignore_append_global_steps_to_folder:
            curr_ckpt_path = os.path.join(ckpt_path, _DIRECTORY_FORMAT.format(global_steps))
        return curr_ckpt_path

    @classmethod
    def make_ckpt_dirs(
        cls,
        curr_ckpt_path: str,
        ckpt_names: List[str],
        supported_ckpt_types: Set[str] = SUPPORTED_TYPES,
    ):
        # Create all folders for once to avoid multiple collective barriers
        if (
            bfile.get_schema(curr_ckpt_path) == bfile.FileType.LOCAL and int(os.environ.get("LOCAL_RANK", "0")) == 0
        ) or dist.get_rank() == 0:
            for name in ckpt_names:
                if name in supported_ckpt_types:
                    state_path = os.path.join(curr_ckpt_path, name)
                    bfile.makedirs(state_path)
                else:
                    raise ValueError(f"unsupported ckpt types: {name}")
        dist.barrier()

    @classmethod
    def ensure_tensor_on_cpu(cls, value):
        def apply(x):
            if isinstance(x, torch.Tensor):
                x = x.cpu()
            return x

        def dict_list_map_outplace(f: Callable, x: Union[dict, list]):
            """Maps dicts and lists *out-of-place* with a given function."""
            if isinstance(x, dict):
                return {k: dict_list_map_outplace(f, v) for k, v in x.items()}
            elif isinstance(x, list):
                return [dict_list_map_outplace(f, v) for v in x]
            elif isinstance(x, tuple):
                return tuple([dict_list_map_outplace(f, v) for v in x])
            elif isinstance(x, set):
                return {dict_list_map_outplace(f, v) for v in x}
            elif isinstance(x, deque):
                return type(x)([dict_list_map_outplace(f, v) for v in x])
            # TODO: deal with more special data containers.
            else:
                return f(x)

        return dict_list_map_outplace(apply, value)

    @classmethod
    def check_supported_components(cls, checkpoint_state: CheckpointState, supported_types: Set[str]):
        # Check if we support saving /loading the components in checkpoint_state
        for key in checkpoint_state.keys():
            if key not in supported_types:
                raise ValueError(f"{key} is not supported by {cls._framework_name} Checkpointer")
