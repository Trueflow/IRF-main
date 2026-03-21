# CDS predict_net (Intrinsic) 
# https://github.com/lich14/CDS/blob/main/CDS_SMAC/QPLEX-master-SC2/pymarl-master/src/modules/intrinsic/predict_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Distribution, Normal

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# LOG_SIG_MAX = 2 / LOG_SIG_MIN = -20 / epsilon = 1e-6
class Predict_Net(nn.Module): # without id
    def __init__(self, args, lr=1e-3):
        super(Predict_Net, self).__init__()
        self.num_inputs = args.embadding_dim + args.obs_size + args.action_size
        self.hidden_dim = args.embadding_dim # hyperparameter (default : 128)
        self.num_outputs = args.obs_size

        self.linear1 = nn.Linear(self.num_inputs, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.last_fc = nn.Linear(self.hidden_dim, self.num_outputs)

        self.apply(weights_init_)
        self.lr = lr

        self.optimiser = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):
        h = F.relu(self.linear1(input))
        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        return x

    def get_log_pi(self, own_variable, other_variable):
        predict_variable = self.forward(own_variable)
        log_prob = -1 * F.mse_loss(predict_variable, other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def update(self, own_variable, other_variable):
        predict_variable = self.forward(own_variable)
        loss = F.mse_loss(predict_variable, other_variable, reduction='none')
        loss = loss.sum(dim=-1).mean()
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optimiser.step()


class Combined_Predict_Net(nn.Module): # with id, IVF 안들어감. 기본 설정 layer_norm=False
    def __init__(self, args, lr=1e-3):
        super(Combined_Predict_Net, self).__init__()
        self.num_inputs = args.embadding_dim + args.obs_size + args.action_size + args.num_agents
        self.hidden_dim = args.embadding_dim # hyperparameter (default : 128)
        self.n_agents = args.num_agents
        self.num_outputs = args.obs_size

        self.linear1 = nn.Linear(self.num_inputs, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim + self.n_agents, self.hidden_dim)
        self.last_fc = nn.Linear(self.hidden_dim, self.num_outputs)

        self.apply(weights_init_)
        self.lr = lr
        self.optimiser = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input, add_id):
        h = F.relu(self.linear1(input))
        h = torch.cat([h, add_id], dim=-1)
        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        return x

    def get_log_pi(self, own_variable, other_variable, add_id):
        predict_variable = self.forward(own_variable, add_id)
        log_prob = -1 * F.mse_loss(predict_variable, other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def update(self, own_variable, other_variable, add_id):
        predict_variable = self.forward(own_variable, add_id)
        loss = F.mse_loss(predict_variable, other_variable, reduction='none')
        loss = loss.sum(dim=-1).mean()
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optimiser.step()