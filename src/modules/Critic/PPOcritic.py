import torch
import torch.nn as nn
import torch.nn.functional as F

# PPO value network
class PPOcritic(torch.nn.Module):
    def __init__(self, args):
        super(PPOcritic, self).__init__()
        self.args = args
        self.state_size = args.upper_state
        self.n_agents = args.num_agents
        self.device = args.device
        self.n_step = args.n_step
        self.embadding_dim = args.upper_embadding_dim

        self.output_type = "v"

        self.fc1 = nn.Linear(self.state_size, self.embadding_dim)
        self.fc2 = nn.Linear(self.embadding_dim, self.embadding_dim)
        self.fc3 = nn.Linear(self.embadding_dim, 1)

    def forward(self, state):
        x_1 = F.relu(self.fc1(state))
        x_2 = F.relu(self.fc2(x_1))
        v = F.relu(self.fc3(x_2))
        return v