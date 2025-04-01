################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
################################################################################

import os
import time
from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Optional

import torch.distributed as dist

from bytecheckpoint.checkpointer.meta_type import (
    STATE_DICT_TYPE,
)
from bytecheckpoint.engine import _load_engine, _store_engine
from bytecheckpoint.storage import CKPTCounter, read_from_store, write_to_store
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()


def save_extra_state(
    file_name: str,
    state_path: str,
    root_path: str,
    state_dict: STATE_DICT_TYPE,
    ckpt_name: str,
    framework_name: str,
    suffix: Optional[str],
    save_ckpt_start_time: float,
    ckpt_counter: CKPTCounter,
    global_steps: Optional[int],
    callback: Optional[Callable],
    async_io: bool = True,
) -> None:
    """
    Save checkpoint states except for the model and optimizer.
    Typically extra state includes random states, learning rate schedulers, number of steps, and the dataloader.

    Args:
        file_name (str): The file name of the persisted extra state.
        state_path (str): The save path of the extra state.
        root_path (str): The root path for the checkpoint.
        state_dict (STATE_DICT_TYPE): A dictionary containing extra state. The keys are strings, and the values are serializable objects like tensors or strings.
        ckpt_name (str): The name of the checkpoint.
        framework_name (str): The name of the framework.
        suffix (Optional[str]): An optional suffix for the checkpoint.
        save_ckpt_start_time (float): The start time of saving the checkpoint.
        ckpt_counter (CKPTCounter): The checkpoint counter.
        global_steps (Optional[int]): The global steps.
        callback (Optional[Callable]): An optional callback function.
        async_io (bool, optional): A boolean indicating whether to use asynchronous file I/O and serialization when writing files to the local path. Defaults to True.

    Returns:
        None
    """
    logger.info("[Rank = %s] Start storing extra states.", dist.get_rank())

    # Register resources lazily.
    assert _store_engine.register_resources(ckpt_name, framework_name, suffix, state_path)

    # Wait for last write futures to finish.
    store_extra_local_start_time = time.time()
    write_futures = _store_engine.execute(
        ckpt_name,
        framework_name,
        suffix=suffix,
        file_name=file_name,
        state_dict=state_dict,
        async_io=async_io,
    )
    store_extra_cost_time = time.time() - store_extra_local_start_time
    logger.info("[Rank = %s] Finish storing extra states. Time cost: %s s", dist.get_rank(), store_extra_cost_time)

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


def load_extra_state(
    file_name: str,
    path: str,
    ckpt_name: str,
    framework_name: str,
    suffix: Optional[str],
    use_pickle: bool = False,
    allow_not_exists: bool = False,
    async_io: bool = False,
) -> List[Future[Dict[str, Any]]]:
    """
    Load checkpoint states except for the model and optimizer.
    Typically extra state includes random states, learning rate schedulers, number of steps, and the dataloader.

    Args:
        file_name (str): The file name of the persisted extra state.
        path (str): The load path of the extra state.
        ckpt_name (str): The name of the checkpoint.
        framework_name (str): The name of the framework.
        suffix (Optional[str]): An optional suffix for the checkpoint.
        use_pickle (bool, optional): A boolean indicating whether to use pickle for serialization. Defaults to False.
        allow_not_exists (bool, optional): A boolean indicating whether to ignore loading extra state when the file does not exist. Defaults to False.
        async_io (bool, optional): A boolean indicating whether to use asynchronous file I/O. Defaults to False.

    Returns:
        List[Future[Dict[str, Any]]]: A list of futures that will resolve to dictionaries containing extra states.
    """

    # Check if the file exists.
    if not file_exists(path, file_name):
        if allow_not_exists:
            logger.warning(
                "[Rank = %s] The target file for loading %s does not exist. Will return None.",
                dist.get_rank(),
                file_name,
            )
            future = Future()
            future.set_result(dict())
            if not async_io:
                return future.result()
            else:
                return future
        else:
            raise FileNotFoundError(f"{file_name} does not exist.")

    # Read from storage.
    if not isinstance(file_name, List):
        file_name = [file_name]
    read_futures = read_from_store(
        ckpt_name,
        framework_name,
        suffix,
        file_name,
        path,
        async_io=True,
    )

    # Load from file.
    assert _load_engine.register_resources(ckpt_name, framework_name, suffix)
    load_futures = _load_engine.execute(
        ckpt_name,
        framework_name,
        suffix,
        read_futures,
        use_pickle=use_pickle,
    )
    assert len(load_futures) == 1, "Only one file is expected for loading extra state."
    return load_futures


def file_exists(path: str, file_name: str) -> bool:
    """
    Check if the file exists.

    Args:
        path (str): The load path of the extra state.
        file_name (str): The file name of the persisted extra state.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    # Check if the file exists.
    file_path = os.path.join(path, file_name)
    return os.path.exists(file_path)
