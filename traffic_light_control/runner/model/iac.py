from typing import Dict, List

import torch.nn as nn
import torch.nn.functional as F


def make_iac_model(config: Dict, agents_id: List[str]):
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
