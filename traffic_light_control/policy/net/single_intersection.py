
import torch.nn as nn
import torch.nn.functional as F


class SingleIntesection(nn.Module):
    def __init__(self, input_space, output_space) -> None:
        super(SingleIntesection, self).__init__()
        self.fc1 = nn.Linear(input_space, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, output_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)

        return action
