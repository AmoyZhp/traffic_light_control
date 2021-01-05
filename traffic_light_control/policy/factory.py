
from typing import List
from policy.algorithm.actor_critic import ActorCritic
import policy.buffer as buffer
import policy.net as net
from policy.algorithm.vdn import VDN
from policy.algorithm.dqn import DQN
from policy.wrapper.ctde_wrapper import CTDEWrapper
from policy.wrapper.independent_wrapper import IndependentWrapper


def get_policy(id_, config):
    if id_ == "VDN":
        return __get_VDN(config)
    elif id_ == "IQL":
        return __get_IQL(config)
    elif id_ == "IAC":
        return __get_AC(config)
    else:
        print("invalid id {}".format(id_))


def __get_IQL(config):

    local_ids: List = config["local_ids"]
    buffer_config = config["buffer"]
    mode = config["mode"]
    policies = {}
    buffers = {}

    for id_ in local_ids:
        net_id = config["net_id"]
        acting_net = net.get_net(net_id, config)
        target_net = net.get_net(net_id, config)
        policies[id_] = DQN(
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
        if mode == "train":
            buffers[id_] = buffer.get_buffer(
                buffer_config["id"],  buffer_config)
    policy_wrapper = IndependentWrapper(
        policies=policies,
        buffers=buffers,
        local_ids=local_ids,
        batch_size=config["batch_size"],
        mode=mode
    )
    return policy_wrapper


def __get_VDN(config):
    net_id = config["net_id"]
    local_ids = config["local_ids"]
    acting_nets = {}
    target_nets = {}
    for id_ in local_ids:
        acting_nets[id_] = net.get_net(net_id, config)
        target_nets[id_] = net.get_net(net_id, config)
    buffer_config = config["buffer"]
    mode = config["mode"]
    policy_ = VDN(
        local_ids=local_ids,
        acting_nets=acting_nets,
        target_nets=target_nets,
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

    buffer_ = buffer.get_buffer(buffer_config["id"], buffer_config)
    wrapper = CTDEWrapper(
        policy_=policy_,
        buffer_=buffer_,
        local_ids=local_ids,
        batch_size=config["batch_size"],
        mode=mode
    )
    return wrapper


def __get_AC(config):

    local_ids: List = config["local_ids"]
    buffer_config = config["buffer"]
    mode = config["mode"]
    policies = {}
    buffers = {}

    for id_ in local_ids:
        net_id = config["net_id"]
        actor_net = net.get_net("IActor", config)
        critic = net.get_net(net_id, config)
        policies[id_] = ActorCritic(
            actor_net=actor_net,
            critic_net=critic,
            learning_rate=config["learning_rate"],
            discount_factor=config["discount_factor"],
            device=config["device"],
            action_space=config["output_space"],
            state_space=config["input_space"]
        )
        if mode == "train":
            buffers[id_] = buffer.get_buffer(
                "on_policy_buffer",  buffer_config)
    policy_wrapper = IndependentWrapper(
        policies=policies,
        buffers=buffers,
        local_ids=local_ids,
        batch_size=config["batch_size"],
        mode=mode
    )
    return policy_wrapper
