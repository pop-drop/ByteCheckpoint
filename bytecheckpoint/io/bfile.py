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
# Existing APIs all follow the rule here:
# https://www.tensorflow.org/api_docs/python/tf/io/gfile

import contextlib
import enum
import os
import shutil
import uuid

BFILE_DEFAULT_TIMEOUT = None


class FileType(enum.Enum):
    LOCAL = 0
    HDFS = 1


def get_schema(path: str):
    if path.startswith("hdfs://"):
        return FileType.HDFS
    else:
        return FileType.LOCAL


def rename(src, dst, overwrite=False):
    return os.replace(src, dst)


def list_files(folders, recursive=False):
    files = []
    for folder in folders:
        files.extend(local_list_folder(folder, recursive))
    return files


def local_list_folder(folder_path: str, recursive: bool = False):
    file_paths = []
    if recursive:
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                file_paths.append(file_path)
    else:
        if os.path.isdir(folder_path):
            file_paths.extend([os.path.join(folder_path, d) for d in os.listdir(folder_path)])
        elif os.path.isfile(folder_path):
            file_paths.append(folder_path)

    return file_paths


def listdir(path):
    absolute_files = list_files([path])
    return [f[f.rfind("/") + 1 :] for f in absolute_files]


def remove(path):
    shutil.rmtree(path)


def exists(path):
    return os.path.exists(path)


def makedirs(path):
    os.makedirs(path, exist_ok=True)


@contextlib.contextmanager
def BFile(name, mode="r", **kwargs):
    t = get_schema(name)
    if t == FileType.LOCAL:
        with open(name, mode) as f:
            yield f
    else:
        raise NotImplementedError


# ---- Below is some useful utilities -----


def atomic_write(path: str, content: bytes, **kwargs):
    tmp_path = path + "_tmp_" + str(uuid.uuid4())
    with BFile(tmp_path, "wb", **kwargs) as f:
        f.write(content)
    rename(tmp_path, path, overwrite=True)


def safe_atomic_write(path: str, content: bytes, **kwargs):
    makedirs(os.path.dirname(path))
    atomic_write(path, content, **kwargs)


def is_local_path(path: str):
    t = get_schema(path)
    if t == FileType.LOCAL:
        return True
    return False
