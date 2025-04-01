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
import time

import torch.distributed as dist

from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger
from bytecheckpoint.utilities.server import server_lib

logger = get_bytecheckpoint_logger()

_addr = None
_stub = None


def setup_rpc_service():
    global _addr, _stub
    # Launch RPC service, one time cost, don't take it into saving time cost.
    if _addr is None or _stub is None:
        rpc_start_time = time.time()
        obj_list = [None]
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            obj_list[0] = server_lib.start_server_in_new_process(world_size)
            logger.info("Start rpc coordinator server on %s", obj_list[0])
        dist.broadcast_object_list(obj_list)
        _addr = obj_list[0]
        _stub = server_lib.get_stub(_addr)
        rpc_cost_time = time.time() - rpc_start_time
        logger.info("Finish rpc start. Time cost: %s s", rpc_cost_time)


def get_stub():
    if _stub is None:
        logger.warning("rpc service is not initialized yet")
    return _stub
