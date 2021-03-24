import logging
from envs.factory import make
from envs.factory import get_default_config_for_single
from envs.factory import get_default_config_for_multi

__all__ = [
    "make",
    "get_default_config_for_single",
    "get_default_config_for_multi",
]
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s", )

stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)