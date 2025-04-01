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

import json
import os
from collections import OrderedDict

import safetensors.torch
from safetensors.torch import load_file

from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()


def save_sharded_hf_checkpoint(state_dict: dict, save_dir: str, max_shard_size: int = 5 * 1024**3):
    shards = []
    current_shard = OrderedDict()
    current_size = 0

    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        tensor_size = tensor.numel() * tensor.element_size()  # Size in bytes

        # Start a new shard if adding this tensor exceeds the max size
        if current_size + tensor_size > max_shard_size and len(current_shard) > 0:
            shards.append(current_shard)
            current_shard = OrderedDict()
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    # Add the last shard
    if current_shard:
        shards.append(current_shard)

    # Save shards and build index
    index = {"metadata": {}, "weight_map": {}}
    total_size = 0

    for shard_idx, shard in enumerate(shards):
        shard_file = f"model-{shard_idx + 1:05d}-of-{len(shards):05d}.safetensors"
        shard_path = os.path.join(save_dir, shard_file)

        # Save shard to safetensors
        safetensors.torch.save_file(shard, shard_path)
        # Update index
        for key in shard:
            index["weight_map"][key] = shard_file

        # Calculate total size (optional)
        total_size += sum(t.numel() * t.element_size() for t in shard.values())

    # Add metadata (customize as needed)
    index["metadata"]["total_size"] = total_size

    # Save index.json
    index_path = os.path.join(save_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return total_size


def load_hf_checkpoint_to_state_dict(load_dir: str):
    index_file_name = os.path.join(load_dir, "model.safetensors.index.json")

    with open(index_file_name) as f:
        index_data = json.load(f)

    # Get all safetensor files
    safetensor_files = set(index_data["weight_map"].values())

    # Load all weights
    model_weights = {}
    for file in safetensor_files:
        full_path = os.path.join(load_dir, file)
        print(f"Loading {full_path}...")
        model_weights.update(load_file(full_path))

    return model_weights
