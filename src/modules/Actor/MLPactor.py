import torch.nn
import torch.nn.functional as F

class MLPactor(torch.nn.Module):
    def __init__(self, args):
        super(MLPactor, self).__init__()
        self.device = args.device
        self.state_size = args.state_size
        self.action_size = args.action_size
        self.embadding_dim = args.embadding_dim

        self.d1 = torch.nn.Linear(self.state_size, self.embadding_dim).to(self.device)
        self.d2 = torch.nn.Linear(self.embadding_dim, self.embadding_dim).to(self.device)
        self.pi = torch.nn.Linear(self.embadding_dim, self.action_size).to(self.device)

    def forward(self, inputs):
        x1 = F.relu(self.d1(inputs))
        x2 = F.relu(self.d2(x1))
        q = self.pi(x2)
        return F.softmax(q, dim=-1)