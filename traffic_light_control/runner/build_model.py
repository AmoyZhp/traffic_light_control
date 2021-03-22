import logging
import torch

import torch.nn as nn
import torch.nn.functional as F

import hprl

logger = logging.getLogger(__package__)


def build_model(
    policy: hprl.PolicyTypes,
    env: hprl.MultiAgentEnv,
):
    agents_id = env.get_agents_id()
    embed_dim = max(len(agents_id) * 4, 32)
    model_config = {
        "central_state": env.get_central_state_space(),
        "local_state": env.get_local_state_space(),
        "central_action": env.get_central_action_space(),
        "local_action": env.get_local_action_space(),
        "embed_dim": embed_dim,
    }
    models = _make_model(
        policy_type=policy,
        config=model_config,
        agents_id=agents_id,
    )
    return models


def _make_model(policy_type, config, agents_id):
    if policy_type == hprl.PolicyTypes.IQL:
        return _make_iql_model(config, agents_id)
    elif (policy_type == hprl.PolicyTypes.IAC
          or policy_type == hprl.PolicyTypes.PPO):
        return _make_ac_model(config, agents_id)
    elif policy_type == hprl.PolicyTypes.VDN:
        return _make_vdn_model(config, agents_id)
    elif policy_type == hprl.PolicyTypes.COMA:
        return _make_coma_model(config, agents_id)
    elif policy_type == hprl.PolicyTypes.QMIX:
        return _make_qmix_model(config, agents_id)
    else:
        raise ValueError("invalid trainer type {}".format(policy_type))


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


def _make_qmix_model(config, agents_id):
    actors_net = {}
    actors_target_net = {}
    for id in agents_id:
        actors_net[id] = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )
        actors_target_net[id] = ICritic(
            input_space=config["local_state"][id],
            output_space=config["local_action"][id],
        )
    critic_net = QMixer(
        state_dim=config["central_state"],
        embed_dim=config["embed_dim"],
        n_agents=len(agents_id),
    )
    critic_target_net = QMixer(
        state_dim=config["central_state"],
        embed_dim=config["embed_dim"],
        n_agents=len(agents_id),
    )
    model = {
        "critic_net": critic_net,
        "critic_target_net": critic_target_net,
        "actors_net": actors_net,
        "actors_target_net": actors_target_net,
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
    critic_input_space = (config["central_state"] + state_space +
                          len(agents_id) + len(agents_id) * action_space)
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


class ICritic(nn.Module):
    def __init__(self, input_space, output_space) -> None:
        super(ICritic, self).__init__()
        self.fc1 = nn.Linear(input_space, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = self.fc4(x)
        return action


class IActor(nn.Module):
    def __init__(self, input_space, output_space) -> None:
        super(IActor, self).__init__()
        self.fc1 = nn.Linear(input_space, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = F.softmax(self.fc3(x), dim=-1)
        return action


class COMACritic(nn.Module):
    def __init__(self, input_space, output_space) -> None:
        super(COMACritic, self).__init__()
        self.fc1 = nn.Linear(input_space, 1028)
        self.fc2 = nn.Linear(1028, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, output_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        action = self.fc6(x)
        return action


class QMixer(nn.Module):
    def __init__(self, state_dim, embed_dim, n_agents) -> None:
        super(QMixer, self).__init__()
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.n_agents = n_agents

        self.weight_layer_1 = nn.Linear(
            self.state_dim,
            self.embed_dim * self.n_agents,
        )
        self.bias_layer_1 = nn.Linear(state_dim, embed_dim)

        self.weight_layer_2 = nn.Linear(self.state_dim, self.embed_dim)
        self.bias_layer_2 = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, joint_q_vals, state):
        # joint q vals : Tensor of shape [B, n_agents, 1]
        # state : Tensor of shape [B, state_space]
        joint_q_vals = joint_q_vals.view(-1, 1, self.n_agents)

        weight_1 = torch.abs(self.weight_layer_1(state))
        bias_1 = self.bias_layer_1(state)

        weight_1 = weight_1.view(-1, self.n_agents, self.embed_dim)
        bias_1 = bias_1.view(-1, 1, self.embed_dim)

        hidden_1 = torch.bmm(joint_q_vals, weight_1) + bias_1
        hidden_1 = nn.functional.elu(hidden_1)
        # hidden shape is [B, 1, embed_dim]

        weight_2 = torch.abs(self.weight_layer_2(state))
        bias_2 = self.bias_layer_2(state)

        weight_2 = weight_2.view(-1, self.embed_dim, 1)
        bias_2 = bias_2.view(-1, 1, 1)

        output = torch.bmm(hidden_1, weight_2) + bias_2
        output = output.view(-1, 1)
        return output