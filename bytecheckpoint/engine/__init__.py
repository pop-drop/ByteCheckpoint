import atexit

from .load_engine import LoadEngine
from .store_engine import StoreEngine

_store_engine = StoreEngine()
_load_engine = LoadEngine()

atexit.register(_store_engine.cleanup_resources)
atexit.register(_load_engine.cleanup_resources)
