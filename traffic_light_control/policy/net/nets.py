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
        self.fc1 = nn.Linear(input_space, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = F.softmax(self.fc4(x), dim=-1)
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
