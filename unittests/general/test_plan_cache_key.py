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
import torch

from bytecheckpoint.planner.default_planner import PlanCacheKey, create_plan_cache_key

if __name__ == "__main__":
    state_dict_0 = {
        "model": torch.rand((10, 10), dtype=torch.float32),
        "optimizer": torch.rand((10, 10), dtype=torch.float32),
        "extra_state": {"key1": "value1", "key2": 65500, "key3": [1, 2, 3]},
    }
    plan_key_0: PlanCacheKey = create_plan_cache_key(state_dict_0)
    print(plan_key_0)

    state_dict_1 = {
        "model": torch.ones((10, 10), dtype=torch.float32),
        "optimizer": torch.zeros((10, 10), dtype=torch.float32),
        "extra_state": {"key1": "value1", "key2": 65500, "key3": [1, 2, 3]},
    }

    plan_key_1: PlanCacheKey = create_plan_cache_key(state_dict_1)
    print(plan_key_1)
    assert plan_key_0 == plan_key_1, "Should be equal"

    state_dict_2 = {
        "new_key": torch.zeros((10, 10)),
        "model": torch.ones((10, 10), dtype=torch.float32),
        "optimizer": torch.zeros((10, 10), dtype=torch.float32),
        "extra_state": {"key1": "value1", "key2": 65500, "key3": [1, 2, 3]},
    }

    plan_key_2: PlanCacheKey = create_plan_cache_key(state_dict_2)
    print(plan_key_2)
    assert plan_key_0 != plan_key_2, "Should NOT be equal"

    state_dict_3 = {
        "model": torch.ones((10, 10), dtype=torch.float32),
        "optimizer": torch.zeros((10, 10), dtype=torch.float32),
        "extra_state": {"key1": "value1", "key2": 65536, "key3": [1, 2, 3]},
    }
    plan_key_3: PlanCacheKey = create_plan_cache_key(state_dict_3)
    assert plan_key_0 == plan_key_3, "Should be equal"

    print("Key test pass")
