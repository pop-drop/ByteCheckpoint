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

import argparse
import os
import time

import torch.distributed as dist

from bytecheckpoint.utilities.ckpt_format.merge_tool import bytecheckpoint_ckpt_to_pytorch_ckpt


def setup(rank, world_size, enable_nccl=False):
    backend = "gloo"
    if enable_nccl:
        backend = "nccl"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class MockedModel:
    def __init__(self, state_dict) -> None:
        self.state_dict_ = state_dict

    def state_dict(self):
        return self.state_dict_


class MockedOptimizer:
    def __init__(self, state_dict) -> None:
        self.state_dict_ = state_dict

    def state_dict(self):
        return self.state_dict_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=str, default="fsdp")
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--model_only", action="store_true")
    parser.add_argument("--optimizer_only", action="store_true")
    parser.add_argument("--enable_nccl", action="store_true")

    # Load ckpt into the memory
    args = parser.parse_args()
    begin = time.time()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(rank=rank, world_size=world_size, enable_nccl=args.enable_nccl)
    print(f"my_rank {dist.get_rank()} my world size = {dist.get_world_size()}")

    state_dict = bytecheckpoint_ckpt_to_pytorch_ckpt(
        args.ckpt_path,
        args.output_path,
        args.framework,
        model_only=args.model_only,
        optimizer_only=args.optimizer_only,
        return_dict=True,
    )

    if not args.optimizer_only:
        print(state_dict)
        model_sd = state_dict["model"]
    if not args.model_only:
        optimizer_sd = state_dict["optimizer"]
    # Get model sd, if you want to get optimizer_sd,
    # optimizer_sd = state_dict['optimizer']
    # Add your own logics to modify the checkpoint.
    # The state dict contains full tensors in the entire model (not sharded)
    # Example: double experts
    """

    for key in list(model_sd.keys()):
        if 'moe.experts.fc1_2' in key or 'moe.experts.fc1_1' in key:
            doubled_experts = torch.cat([model_sd[key], model_sd[key]], dim=0)
            model_sd[key] = doubled_experts

    for key in list(optimizer_sd.keys()):
        if 'moe.experts.fc1_2' in key or 'moe.experts.fc1_1' in key:
            doubled_experts = torch.cat([optimizer_sd[key], optimizer_sd[key]], dim=0)
            optimizer_sd[key] = doubled_experts
    """
    # Your own code:

    from bytecheckpoint import FSDP2Checkpointer

    ckpt_state = {}
    if not args.optimizer_only:
        ckpt_state = {"model": MockedModel(model_sd)}
    if not args.model_only:
        ckpt_state["optimizer"] = MockedOptimizer(optimizer_sd)
    FSDP2Checkpointer.save(path=args.output_path, checkpoint_state=ckpt_state, modifying_ckpt=True)

    cleanup()
