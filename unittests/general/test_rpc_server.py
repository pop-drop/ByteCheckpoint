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
import threading
import unittest

from bytecheckpoint.utilities.server import server_lib


def create_server(world_size):
    addr = server_lib.start_server_in_new_process(world_size)
    return server_lib.get_stub(addr)


class ServerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._world_size = 5
        self._stub = create_server(self._world_size)

    def test_gather(self):
        call_results = [None for _ in range(self._world_size)]

        def gather(i):
            # Simulate race.
            for retries in range(10):
                call_results[i] = server_lib.gather(self._stub, gather_rank=0, rank=i, obj=i)

        client_ths = []
        for i in range(self._world_size):
            th = threading.Thread(target=gather, args=(i,))
            th.start()
            client_ths.append(th)
        [th.join() for th in client_ths]

        self.assertListEqual(call_results[0], list(range(0, self._world_size)))
        d = server_lib.get_server_status(self._stub)
        self.assertDictEqual(d["gather_dict"], {})

    def test_broadcast(self):
        call_results = [None for _ in range(self._world_size)]

        def broadcast(i):
            # Simulate race
            for retries in range(10):
                call_results[i] = server_lib.broadcast(self._stub, src_rank=0, rank=i, obj=i)

        client_ths = []
        for i in range(self._world_size):
            th = threading.Thread(target=broadcast, args=(i,))
            th.start()
            client_ths.append(th)
        [th.join() for th in client_ths]

        for i in range(self._world_size):
            self.assertEqual(call_results[i], 0)

        d = server_lib.get_server_status(self._stub)
        self.assertDictEqual(d["bc_dict"], {})


if __name__ == "__main__":
    unittest.main()
