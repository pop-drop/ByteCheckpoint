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

from bytecheckpoint.utilities.ckpt_format.ckpt_loader import CKPTLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # The checkpoint path contains .metadata
    # For example: /mnt/disk/bytecheckpoint/fsdp_fast_ckpt/model
    parser.add_argument("--ckpt_path", required=True, type=str)
    args = parser.parse_args()

    url = args.ckpt_path
    loader = CKPTLoader(url)

    metadata = loader.load_metadata()

    for key, tensor in metadata.state_dict_metadata.items():
        print(f"key={key} value={tensor}")
    print(metadata.user_defined_dict)
