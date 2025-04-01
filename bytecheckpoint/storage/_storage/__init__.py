import atexit

from .base_storage import CKPTCounter
from .local_storage import LocalStorageReader, LocalStorageWriter

_local_storage_writer = LocalStorageWriter()
_local_storage_reader = LocalStorageReader()


atexit.register(_local_storage_writer.cleanup_resources)
atexit.register(_local_storage_reader.cleanup_resources)
