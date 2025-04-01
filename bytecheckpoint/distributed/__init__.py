from ..utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()

try:
    from .fsdp_wrapper import _FSDPTreeTopoWrapper
except ImportError as e:
    logger.debug("_FSDPTreeTopoWrapper not imported. Ignore this warning if you are not using FSDP")
except NameError as e:
    logger.debug("_FSDPTreeTopoWrapper not imported. Ignore this warning if you are not using FSDP")
