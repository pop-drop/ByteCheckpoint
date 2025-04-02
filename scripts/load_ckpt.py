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
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from bytecheckpoint.utilities.ckpt_format.ckpt_loader import distcp_load_tool

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # The checkpoint path contains .metadata
    # For example: /mnt/disk/bytecheckpoint/fsdp_fast_ckpt/model
    parser.add_argument("--ckpt_path", required=True, type=str)
    # The file to read,
    # For example: __0_0.distcp
    parser.add_argument("--file_path", required=True, type=str)
    args = parser.parse_args()

    state_dict = distcp_load_tool(args.ckpt_path, args.file_path)

    for key, offset_tensor_list in state_dict.items():
        print(f"key={key} ")
        for kv_pair in offset_tensor_list:
            print(f"offset={kv_pair['offset']}, tensor_or_object={kv_pair['tensor_or_object']}")

# python3 scripts/load_ckpt.py --ckpt_path tmp_checkpoint_dir_fsdp/global_step_0/optimizer --file_path __0_0.distcp
