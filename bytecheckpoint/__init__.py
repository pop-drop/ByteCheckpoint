from enum import Enum

import torch

from .config import BYTECHECKPOINT_GLOBAL_CONFIG
from .version import __version__


def _get_device_module(device_type: str):
    device_module = getattr(torch, device_type, None)
    if device_module is None:
        raise RuntimeError(
            f"Device '{device_type}' does not have a corresponding module registered as 'torch.{device_type}'."
        )
    return device_module


if not hasattr(torch._utils, "_get_device_module"):
    torch._utils._get_device_module = _get_device_module


class _MEM_FORMAT_ENCODING(Enum):
    """Describe the memory format of a tensor."""

    TORCH_CONTIGUOUS_FORMAT = 0
    TORCH_CHANNELS_LAST = 1
    TORCH_PRESERVE_FORMAT = 2


from torch.distributed._shard.sharded_tensor.metadata import MEM_FORMAT_ENCODING
from torch.distributed.checkpoint import metadata

if not hasattr(metadata, "_MEM_FORMAT_ENCODING"):
    metadata._MEM_FORMAT_ENCODING = MEM_FORMAT_ENCODING

from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()

# Usage 1: Use load/save API.
from .api.load import load
from .api.save import save

# Usage 2: Use checkpointer.
from .checkpointer.ddp_checkpointer import DDPCheckpointer
from .checkpointer.fsdp_checkpointer import FSDPCheckpointer

try:
    from .checkpointer.fsdp2_checkpointer import FSDP2Checkpointer
except Exception as e:
    logger.warning("Failed to import FSDP2Checkpointer.")
    pass
