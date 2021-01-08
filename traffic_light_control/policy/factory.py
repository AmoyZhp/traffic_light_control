
from typing import List
from policy.algorithm.actor_critic import ActorCritic
from policy.algorithm.coma import COMA
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
    elif id_ == "IQL_PS":
        return __get_IQL_PS(config)
    elif id_ == "IAC":
        return __get_IAC(config)
    elif id_ == "IAC_PS":
        return __get_IAC_PS(config)
    elif id_ == "COMA":
        return __get_COMA(config)
    else:
        print("invalid id {}".format(id_))


def __get_IQL_PS(config):

    local_ids: List = config["local_ids"]
    buffer_config = config["buffer"]
    mode = config["mode"]
    policies = {}
    buffers = {}

    net_id = config["net_id"]
    net_conf = {
        "input_space": config["obs_space"],
        "output_space": config["action_space"],
    }
    acting_net = net.get_net(net_id, net_conf)
    target_net = net.get_net(net_id, net_conf)
    policy_ = DQN(
        acting_net=acting_net,
        target_net=target_net,
        learning_rate=config["learning_rate"],
        discount_factor=config["discount_factor"],
        eps_init=config["eps_init"],
        eps_min=config["eps_min"],
        eps_frame=config["eps_frame"],
        update_period=config["update_period"],
        device=config["device"],
        action_space=config["action_space"],
        state_space=config["obs_space"]
    )
    for id_ in local_ids:
        policies[id_] = policy_
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


def __get_IQL(config):

    local_ids: List = config["local_ids"]
    buffer_config = config["buffer"]
    mode = config["mode"]
    policies = {}
    buffers = {}
    net_conf = {
        "input_space": config["obs_space"],
        "output_space": config["action_space"],
    }
    for id_ in local_ids:
        net_id = config["net_id"]
        acting_net = net.get_net(net_id, net_conf)
        target_net = net.get_net(net_id, net_conf)
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
            action_space=config["action_space"],
            state_space=config["obs_space"]
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
    net_conf = {
        "input_space": config["obs_space"],
        "output_space": config["action_space"],
    }
    for id_ in local_ids:
        acting_nets[id_] = net.get_net(net_id, net_conf)
        target_nets[id_] = net.get_net(net_id, net_conf)
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
        action_space=config["action_space"],
        state_space=config["obs_space"]
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


def __get_IAC(config):

    local_ids: List = config["local_ids"]
    buffer_config = config["buffer"]
    mode = config["mode"]
    policies = {}
    buffers = {}
    net_conf = {
        "input_space": config["obs_space"],
        "output_space": config["action_space"],
    }
    for id_ in local_ids:
        net_id = config["net_id"]
        actor_net = net.get_net("IActor", net_conf)
        critic = net.get_net(net_id, net_conf)
        policies[id_] = ActorCritic(
            actor_net=actor_net,
            critic_net=critic,
            learning_rate=config["learning_rate"],
            discount_factor=config["discount_factor"],
            device=config["device"],
            action_space=config["action_space"],
            state_space=config["obs_space"]
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


def __get_IAC_PS(config):

    local_ids: List = config["local_ids"]
    buffer_config = config["buffer"]
    mode = config["mode"]
    policies = {}
    buffers = {}
    net_id = config["net_id"]
    net_conf = {
        "input_space": config["obs_space"],
        "output_space": config["action_space"],
    }
    actor_net = net.get_net("IActor", net_conf)
    critic = net.get_net(net_id, net_conf)
    policiy_ = ActorCritic(
        actor_net=actor_net,
        critic_net=critic,
        learning_rate=config["learning_rate"],
        discount_factor=config["discount_factor"],
        device=config["device"],
        action_space=config["action_space"],
        state_space=config["obs_space"]
    )
    for id_ in local_ids:
        policies[id_] = policiy_
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


def __get_COMA(config):

    local_ids: List = config["local_ids"]
    critic_input_space = (config["state_space"]
                          + config["obs_space"] + len(local_ids)
                          + len(local_ids) * config["action_space"])
    critic_net_conf = {
        "input_space": critic_input_space,
        "output_space": config["action_space"],
    }
    actor_net_conf = {
        "input_space": config["obs_space"],
        "output_space": config["action_space"],
    }
    critic_net = net.get_net(
        "COMACritic", critic_net_conf)
    act_net = net.get_net(
        "COMAActor", actor_net_conf)

    actor_nets = {}
    for id_ in local_ids:
        actor_nets[id_] = act_net
    policy_ = COMA(
        local_ids=local_ids,
        critic_net=critic_net,
        actor_nets=actor_nets,
        learning_rate=config["learning_rate"],
        discount_factor=config["discount_factor"],
        device=config["device"],
        action_space=config["action_space"],
        state_space=config["state_space"],
        local_obs_space=config["obs_space"]
    )
    buffer_ = buffer.get_buffer(
        "on_policy_buffer",
        config["buffer"]
    )
    wrapper = CTDEWrapper(
        policy_=policy_,
        buffer_=buffer_,
        local_ids=local_ids,
        batch_size=config["batch_size"],
        mode=config["mode"],
    )
    return wrapper
