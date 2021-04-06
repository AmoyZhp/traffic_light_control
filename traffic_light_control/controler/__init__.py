import logging

import hprl
from envs.cityflow import make

from controler.model.iql import make_iql_model
from controler.model.qmix import make_qmix_model
from controler.model.vdn import make_vdn_model
from controler.run import run

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
    entry_point=make_iql_model,
)
hprl.policy.register_model(
    id=hprl.PolicyTypes.VDN.value,
    entry_point=make_vdn_model,
)
hprl.policy.register_model(
    id=hprl.PolicyTypes.QMIX.value,
    entry_point=make_qmix_model,
)
