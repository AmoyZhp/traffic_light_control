import torch.nn as nn
import torch.nn.functional as F
from runner.model.iac import IActor


def make_coma_model(config, agents_id):
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
