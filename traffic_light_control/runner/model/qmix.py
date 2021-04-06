import torch
import torch.nn as nn
from runner.model.iac import ICritic


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
