
import net
from policy.dqn import DQN
from policy.independent_wrapper import IndependentWrapper


def get_policy(id_, config):
    if id_ == "DQN":
        return __get_DQN(config)
    else:
        print("invalid id {}".format(id_))


def get_wrapper(id_, config):
    if id_ == "IL":
        return IndependentWrapper(config, config["mode"])
    else:
        print("invalid wrapper id {}".format(id_))


def __get_DQN(config):
    net_id = config["net_id"]
    acting_net = net.get_net(net_id, config)
    target_net = net.get_net(net_id, config)
    return DQN(
        acting_net=acting_net,
        target_net=target_net,
        learning_rate=config["learning_rate"],
        discount_factor=config["discount_factor"],
        eps_init=config["eps_init"],
        eps_min=config["eps_min"],
        eps_frame=config["eps_frame"],
        update_period=config["update_period"],
        device=config["device"],
        action_space=config["output_space"],
        state_space=config["input_space"]
    )
