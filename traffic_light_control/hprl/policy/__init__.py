from hprl.policy.core import Policy
from hprl.policy.dqn import DQN
from hprl.policy.epsilon_greedy import EpsilonGreedy
from hprl.policy.independent_learner_wrapper import ILearnerWrapper
from hprl.policy.actor_critic import ActorCritic
from hprl.policy.ppo import PPO
from hprl.policy.vdn import VDN
from hprl.policy.coma import COMA
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)

logger.addHandler(handler)

__all__ = [
    "Policy",
    "DQN",
    "PPO",
    "VDN",
    "COMA",
    "ActorCritic",
    "EpsilonGreedy",
    "ILearnerWrapper",
]
