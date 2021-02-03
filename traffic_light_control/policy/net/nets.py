import torch.nn as nn
import torch.nn.functional as F


class SingleIntesection(nn.Module):
    def __init__(self, input_space, output_space) -> None:
        super(SingleIntesection, self).__init__()
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


class VDNLocal(nn.Module):
    def __init__(self, input_space, output_space) -> None:
        super(VDNLocal, self).__init__()
        self.fc1 = nn.Linear(input_space, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = self.fc4(x)
        return action


class COMACritic(nn.Module):
    def __init__(self, input_space, output_space) -> None:
        super(COMACritic, self).__init__()
        self.fc1 = nn.Linear(input_space, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, output_space)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        x4 = F.relu(self.fc4(x3))
        state_action_value = self.fc5(x4)
        return state_action_value


class COMAActor(nn.Module):
    def __init__(self, input_space, output_space) -> None:
        super(COMAActor, self).__init__()
        self.fc1 = nn.Linear(input_space, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_space)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        action = F.softmax(self.fc3(x2), dim=-1)
        return action
