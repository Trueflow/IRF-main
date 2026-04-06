import torch.nn
import torch.nn.functional as F

class RNNActor(torch.nn.Module):
    def __init__(self, args, input_size, action_size):
        super(RNNActor, self).__init__()
        self.device = args.device
        self.n_agents = args.num_agents
        self.embadding_dim = args.embadding_dim
        self.action_size = action_size

        self.fc1 = torch.nn.Linear(input_size, self.embadding_dim)
        self.rnn = torch.nn.GRU(self.embadding_dim, self.embadding_dim) 
        self.fc2 = torch.nn.Linear(self.embadding_dim, self.action_size)
        
        self.to(self.device)
        self.rnn.flatten_parameters()

    def init_hidden(self): # default
        return torch.zeros(1, self.n_agents, self.embadding_dim, device=self.device)

    def init_hidden_2(self):
        return torch.zeros(1, 1, self.embadding_dim, device=self.device)

    def forward(self, inputs, hidden_state):
        if inputs.dim()>3:
            inputs = inputs.squeeze(0)
        x = F.relu(self.fc1(inputs))
        output, h = self.rnn(x, hidden_state) # h is final hidden_state
        q = self.fc2(output)
        q = torch.softmax(q, dim=-1)
        return q, h
    
    def inference_forward(self, inputs, hidden_state):
        with torch.no_grad():
            x = F.relu(self.fc1(inputs))
            output, h = self.rnn(x, hidden_state)
            q = torch.softmax(self.fc2(output), dim=-1)
        del output, x
        return q, h

    def cds_forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        output, _ = self.rnn(x, hidden_state) # output : hidden_state of all timesteps
        q = self.fc2(output)
        q = torch.softmax(q, dim=-1)
        return q, output

    def flattenParameters(self):
        self.rnn.flatten_parameters()


