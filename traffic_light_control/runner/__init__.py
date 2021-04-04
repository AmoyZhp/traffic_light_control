import hprl
from runner.run import run
from runner.build_model import _make_iql_model
import logging
from envs.cityflow import make

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

__all__ = ["run"]

hprl.env.register(id="CityFlow", entry_point=make)
hprl.policy.register_model(
    id=hprl.PolicyTypes.IQL.value,
    entry_point=_make_iql_model,
)
