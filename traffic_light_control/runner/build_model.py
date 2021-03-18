import hprl
from runner.nets import IActor, ICritic
from runner.nets import COMACritic

import logging
logger = logging.getLogger(__package__)


def build_model(args, env):
    agents_id = env.get_agents_id()
    model_config = {
        "central_state": env.get_central_state_space(),
        "local_state": env.get_local_state_space(),
        "central_action": env.get_central_action_space(),
        "local_action": env.get_local_action_space(),
    }
    models = _make_model(
        args.trainer,
        model_config,
        agents_id,
    )
    return models


def _make_model(trainer_type, config, agents_id):
    if trainer_type == hprl.TrainnerTypes.IQL:
        return _make_iql_model(config, agents_id)
    elif trainer_type == hprl.TrainnerTypes.IQL_PS:
        return _make_iql_ps_model(config, agents_id)
    elif (trainer_type == hprl.TrainnerTypes.IAC
          or trainer_type == hprl.TrainnerTypes.PPO):
        return _make_ac_model(config, agents_id)
    elif (trainer_type == hprl.TrainnerTypes.PPO_PS):
        return _make_ppo_ps_model(config, agents_id)
    elif trainer_type == hprl.TrainnerTypes.VDN:
        return _make_vdn_model(config, agents_id)
    elif trainer_type == hprl.TrainnerTypes.COMA:
        return _make_coma_model(config, agents_id)
    elif trainer_type == hprl.TrainnerTypes.IQL_PER:
        return _make_iql_model(config, agents_id)
    else:
        raise ValueError("invalid trainer type {}".format(trainer_type))


def _make_iql_model(config, agents_id):
    models = {}
    for id in agents_id:
        acting_net = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )
        target_net = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )
        models[id] = {
            "acting_net": acting_net,
            "target_net": target_net,
        }
    return models


def _make_iql_ps_model(config, agents_id):
    models = {}
    acting_net = ICritic(
        input_space=config["local_state"][id],
        output_space=config["local_action"][id],
    )
    target_net = ICritic(
        input_space=config["local_state"][id],
        output_space=config["local_action"][id],
    )
    for id in agents_id:
        models[id] = {
            "acting_net": acting_net,
            "target_net": target_net,
        }
    return models


def _make_ac_model(config, agents_id):
    models = {}
    for id in agents_id:
        critic_net = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )

        critic_target_net = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )

        actor_net = IActor(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )

        models[id] = {
            "critic_net": critic_net,
            "critic_target_net": critic_target_net,
            "actor_net": actor_net
        }
    return models


def _make_ppo_ps_model(config, agents_id):
    models = {}
    critic_net = ICritic(
        input_space=config["local_state"][id],
        output_space=config["local_action"][id],
    )

    critic_target_net = ICritic(
        input_space=config["local_state"][id],
        output_space=config["local_action"][id],
    )

    actor_net = IActor(
        input_space=config["local_state"][id],
        output_space=config["local_action"][id],
    )
    for id in agents_id:
        models[id] = {
            "critic_net": critic_net,
            "critic_target_net": critic_target_net,
            "actor_net": actor_net
        }
    return models


def _make_vdn_model(config, agents_id):
    acting_nets = {}
    target_nets = {}
    for id in agents_id:
        acting_nets[id] = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )
        target_nets[id] = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )
    model = {
        "acting_nets": acting_nets,
        "target_nets": target_nets,
    }
    return model


def _make_coma_model(config, agents_id):
    locals_state_space = config["local_state"]
    locals_action_space = config["local_action"]
    for i in range(len(agents_id)):
        if (locals_state_space[agents_id[i]] !=
                locals_state_space[agents_id[i]]):
            raise ValueError(
                "coma only support equal local state space sisutaion")
        if (locals_action_space[agents_id[i]] !=
                locals_action_space[agents_id[i]]):
            raise ValueError("coma only support equal action space")
    action_space = locals_action_space[agents_id[0]]
    state_space = locals_state_space[agents_id[0]]
    print(state_space)
    print(action_space)
    critic_input_space = (config["central_state"] + state_space +
                          len(agents_id) + len(agents_id) * action_space)
    print(critic_input_space)
    critic_net = COMACritic(critic_input_space, action_space)
    target_critic_net = COMACritic(critic_input_space, action_space)
    actors_net = {}
    for id in agents_id:
        actors_net[id] = IActor(state_space, action_space)
    model = {
        "critic_net": critic_net,
        "critic_target_net": target_critic_net,
        "actors_net": actors_net,
    }
    return model