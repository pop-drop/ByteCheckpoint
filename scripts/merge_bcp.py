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
import time

from bytecheckpoint.utilities.ckpt_format.merge_tool import bytecheckpoint_ckpt_to_pytorch_ckpt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=str, default="fsdp")
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--model_only", action="store_true", default=False)
    # Saved decomposed model.
    parser.add_argument("--fsdp_save_decomposed_model", action="store_true", default=False)
    # Export to SafeTensors format.
    parser.add_argument("--safetensors_format", action="store_true", default=False)
    # Add untie embeddings.
    parser.add_argument("--untie_embeddings", action="store_true", default=False)

    args = parser.parse_args()

    begin = time.time()
    print("start to merge bytecheckpoint checkpoint!")

    bytecheckpoint_ckpt_to_pytorch_ckpt(
        args.ckpt_path,
        args.output_path,
        args.framework,
        model_only=args.model_only,
        fsdp_save_decomposed_model=args.fsdp_save_decomposed_model,
        safetensors_format=args.safetensors_format,
        untie_embeddings=args.untie_embeddings,
    )
    print("merge bytecheckpoint checkpoint successfully! cost time:", time.time() - begin, "s")
