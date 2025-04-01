################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
################################################################################

import ctypes
import io
import pickle
from typing import Any, Dict, Tuple

import torch
from torch.serialization import (
    DEFAULT_PROTOCOL,
    StorageType,
    _get_restore_location,
    _maybe_decode_ascii,
    location_tag,
    normalize_storage_type,
)


class _SliceFile(io.IOBase):
    def __init__(self, base_stream: io.IOBase, offset: int, length: int) -> None:
        super().__init__()
        self.base_stream = base_stream
        self.offset = offset
        self.length = length
        # The virtual position of the file slice.
        self.cur_pos = 0
        # Move the file ptr to current position.
        self.base_stream.seek(offset)

    def tell(self):
        return self.cur_pos

    def seek(self, pos, whence=0):
        if whence == 0:
            self.cur_pos = max(0, pos)
            self.base_stream.seek(pos + self.offset, 0)
        elif whence == 1:
            self.cur_pos = max(0, self.cur_pos + pos)
            self.base_stream.seek(self.cur_pos + self.offset + pos, 0)
        elif whence == 2:
            # Pos here is usually a negative number.
            self.cur_pos = max(0, self.length + pos)
            self.base_stream.seek(self.length + self.offset + pos, 0)
        else:
            raise ValueError(f"invalid whence {whence}")
        return self.cur_pos

    def readable(self) -> bool:
        return self.base_stream.readable()

    def seekable(self) -> bool:
        return self.base_stream.seekable()

    def readinto(self, b):
        return self.base_stream.readinto(b)  # type: ignore[attr-defined]

    def read(self, size=-1):
        # Read out the entire file slice.
        if size < 0:
            size = max(0, (self.length - self.cur_pos))
        # Make sure not exceed the remaining slice size.
        else:
            assert size <= self.length - self.cur_pos, (
                f"given size: {size} exceeds the remainin file slice size: {self.length - self.cur_pos}"
            )
        # Update current position before reading.
        self.cur_pos += size
        return self.base_stream.read(size)


def _create_file_slice(file: io.IOBase, offset: int, length: int) -> io.IOBase:
    return _SliceFile(file, offset, length)


def _serialize_tensor(
    tensor: torch.Tensor, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL
) -> Tuple[bytes, bytes]:
    """
    Perform serialization to a single Tensor object. This function separates metadata of tensor from its storage data,
    and return them for further processing.

    Args:
        tensor (torch.Tensor): The torch.Tensor object to be serialized.
        pickle_module (module, optional): The module used for pickling metadata and objects. Defaults to the `pickle` module.
        pickle_protocol (int, optional): The protocol version used for pickling. Can be specified to override the default protocol.

    Returns:
        Tuple[bytes, bytes]: A tuple that includes the metadata and storage data of the input tensor.

    NOTE: This function is implemented based on https://github.com/pytorch/pytorch/blob/a94e507c39df2d2aa8c2ebb70b018e9dda273307/torch/serialization.py#L947.
    """
    serialized_storages = {}
    id_map: Dict[int, str] = {}

    # Since loading storages that view the same data with different dtypes is
    # not supported, we need to keep track of the dtype associated with each
    # storage data_ptr and throw an error if the dtype is ever different.
    # TODO: This feature could be added in the future
    storage_dtypes: Dict[int, torch.dtype] = {}

    def persistent_id(obj):
        """
        Generate a persistent ID for a given object.

        Args:
            obj: The object for which to generate a persistent ID.

        Returns:
            tuple or None: A tuple representing the persistent ID if the object is a storage object, otherwise None.
        """
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = obj._size()

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                storage_numel = storage.nbytes()

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if storage.data_ptr() != 0:
                if storage.data_ptr() in storage_dtypes:
                    if storage_dtype != storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that view the same data as different types"
                        )
                else:
                    storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = id_map.setdefault(storage._cdata, str(len(id_map)))
            location = location_tag(storage)
            serialized_storages[storage_key] = storage

            # Returned persistent_id serves as the reference to retrieve
            # the actual numerial storage data.
            return ("storage", storage_type, storage_key, location, storage_numel)

        return None

    # Prepare the pickle metadata for `Tensor`.
    data_buf = io.BytesIO()
    pickler = pickle_module.Pickler(data_buf, protocol=pickle_protocol)
    pickler.persistent_id = persistent_id
    pickler.dump(tensor)
    metadata = data_buf.getvalue()

    # Prepare the actual storage data for `Tensor`, return the data_ptr and byte length.
    # NOTE: Since we serialize one `Tensor` object each time,
    # the length of `serialized_storages` should always be 1.
    assert len(serialized_storages.keys()) == 1, f"the actual number of keys: {len(serialized_storages.keys())}"
    storage_key = list(serialized_storages.keys())[0]
    storage = serialized_storages[storage_key]
    # The input tensor is already copied to CPU memory.
    assert storage.device.type == "cpu", "input tensor is not copied to"
    # Now that it is on the CPU we can directly copy it into the zip file
    num_bytes = storage.nbytes()
    storagedata = (ctypes.c_char * num_bytes).from_address(storage.data_ptr())
    return metadata, storagedata


def _deserialize_tensor(
    metadata: bytes,
    storagedata: bytes,
    map_location=None,
    pickle_module=pickle,
    **pickle_load_args: Any,
) -> torch.Tensor:
    """
    Perform deserialization to a single Tensor object. This function packs metadata and storagedata to recover the runtime state of
    a tensor object.

    Args:
        metadata (bytes): The metadata of a saved tensor.
        storagedata (bytes): The storage data of a saved tensor.
        map_location (function, torch.device, string, or dict, optional): A function, :class:`torch.device`, string or a dict specifying how to remap storage locations.
        pickle_module (module, optional): The module used for pickling metadata and objects.
        **pickle_load_args (Any, optional): (Python 3 only) Optional keyword arguments passed over to
            :func:`pickle_module.load` and :func:`pickle_module.Unpickler`, e.g.,
            :attr:`errors=...`

    Returns:
        torch.Tensor: A tensor object restored from the metadata and storage data.

    NOTE: This function is implemented based on https://github.com/pytorch/pytorch/blob/a94e507c39df2d2aa8c2ebb70b018e9dda273307/torch/serialization.py#L1613.
    """
    # We input the original metadata for unpickler module to work, but
    # can perform partial read outside the _deserialize_tensor(), and input the already sliced storage data.
    restore_location = _get_restore_location(map_location)

    def load_tensor(dtype, location):
        # Build the storage from given storagedata,
        # we can perform the partial read outside the _deserialize_tensor().
        storage = torch.UntypedStorage.from_buffer(storagedata, dtype=torch.uint8)

        # TODO: Once we decide to break serialization FC, we can
        # stop wrapping with TypedStorage
        typed_storage = torch.storage.TypedStorage(
            wrap_storage=restore_location(storage, location), dtype=dtype, _internal=True
        )
        return typed_storage

    def persistent_load(saved_id):
        # saved_id: ('storage', storage_type, storage_key, location, storage_numel)
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        assert typename == "storage", f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        storage_type, _, location, _ = data
        if storage_type is torch.UntypedStorage:
            dtype = torch.uint8
        else:
            dtype = storage_type.dtype

        typed_storage = load_tensor(dtype, _maybe_decode_ascii(location))

        return typed_storage

    load_module_mapping: Dict[str, str] = {
        # See https://github.com/pytorch/pytorch/pull/51633
        "torch.tensor": "torch._tensor"
    }

    # Need to subclass Unpickler instead of directly monkey-patching the find_class method
    # because it's marked readonly in pickle.
    # The type: ignore is because mypy can't statically determine the type of this class.
    class UnpicklerWrapper(pickle_module.Unpickler):  # type: ignore[name-defined]
        # from https://stackoverflow.com/questions/13398462/unpickling-python-objects-with-a-changed-module-path/13405732
        # Lets us override the imports that pickle uses when unpickling an object.
        # This is useful for maintaining BC if we change a module path that tensor instantiation relies on.
        def find_class(self, mod_name, name):
            if type(name) is str and "Storage" in name:
                try:
                    return StorageType(name)
                except KeyError:
                    pass
            mod_name = load_module_mapping.get(mod_name, mod_name)
            return super().find_class(mod_name, name)

    # Load the data (which may in turn use `persistent_load` to load tensors)
    data_file = io.BytesIO(metadata)

    unpickler = UnpicklerWrapper(data_file, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    result: torch.Tensor = unpickler.load()
    torch._utils._validate_loaded_sparse_tensors()
    return result
