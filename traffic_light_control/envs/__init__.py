import logging
from envs.envs import make_cityflow, make_mp_cityflow
from envs.cityflow import get_default_config_for_single, get_default_config_for_multi

__all__ = [
    "make_cityflow",
    "make_mp_cityflow",
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