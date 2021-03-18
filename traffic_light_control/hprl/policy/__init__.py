from hprl.policy.policy import MultiAgentPolicy
from hprl.policy.single.dqn import DQN
from hprl.policy.decorator.epsilon_greedy import MultiAgentEpsilonGreedy
from hprl.policy.decorator.epsilon_greedy import EpsilonGreedy
from hprl.policy.decorator.independent_learner import IndependentLearner
from hprl.policy.single.actor_critic import ActorCritic
from hprl.policy.single.ppo import PPO
from hprl.policy.multi.vdn import VDN
from hprl.policy.multi.coma import COMA

from hprl.policy.dqn import build_iql_trainer
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
    "IndependentLearner",
    "MultiAgentEpsilonGreedy",
    "build_iql_trainer",
]
