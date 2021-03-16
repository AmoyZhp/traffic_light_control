from hprl.replaybuffer.replay_buffer import ReplayBuffer
from hprl.replaybuffer.common_buffer import CommonBuffer

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)

logger.addHandler(handler)
__all__ = [
    "ReplayBuffer",
    "CommonBuffer",
]
