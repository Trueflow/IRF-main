import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SI_Weight(nn.Module):
    def __init__(self, args):
        super(SI_Weight, self).__init__()

        self.args = args
        self.n_agents = args.num_agents
        self.n_actions = args.action_size
        self.state_dim = int(np.prod(args.state_size))
        self.action_dim = args.num_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim

        self.num_kernel = args.num_kernel

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()

        adv_hypernet_embed = self.args.adv_hypernet_embed
        for i in range(self.num_kernel):  # multi-head attention (default: add_hypernet_layers=1)
            self.key_extractors.append(nn.Linear(self.state_dim, 1))  # key
            self.agents_extractors.append(nn.Linear(self.state_dim, self.n_agents))  # agent
            self.action_extractors.append(nn.Linear(self.state_action_dim, self.n_agents))  # action

    def forward(self, states, actions):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        
        data = th.cat([states, actions], dim=1) # state-action pair

        all_head_key = [k_ext(states) for k_ext in self.key_extractors] # key?
        all_head_agents = [k_ext(states) for k_ext in self.agents_extractors] # agent info
        all_head_action = [sel_ext(data) for sel_ext in self.action_extractors]

        head_attend_weights = []
        for curr_head_key, curr_head_agents, curr_head_action in zip(all_head_key, all_head_agents, all_head_action):
            x_key = F.sigmoid(curr_head_key).repeat(1, self.n_agents) + 1e-10
            # x_key = th.abs(curr_head_key).repeat(1, self.n_agents) + 1e-10
            x_agents = F.sigmoid(curr_head_agents)
            x_action = F.sigmoid(curr_head_action)
            weights = x_key * x_agents * x_action
            head_attend_weights.append(weights)

        head_attend = th.stack(head_attend_weights, dim=1)
        head_attend = head_attend.view(-1, self.num_kernel, self.n_agents)
        head_attend = th.sum(head_attend, dim=1)

        return head_attend
