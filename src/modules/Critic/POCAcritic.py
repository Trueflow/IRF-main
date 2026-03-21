import torch
import torch.nn.functional as F

class POCAcritic(torch.nn.Module):
    def __init__(self, args):
        super(POCAcritic, self).__init__()
        self.device = args.device
        self.state_size = args.state_size
        self.obs_size = args.obs_size
        self.action_size = args.action_size
        self.num_agents = args.num_agents

        self.embadding_dim = args.embadding_dim
        self.transformer_dim = args.embed_dim
        self.num_heads = getattr(args, 'num_heads', 4)
        self.q_d1_dim = args.num_agents * 2 * self.embadding_dim
        
        self.g = torch.nn.ModuleList([torch.nn.Linear(self.state_size+self.obs_size, self.transformer_dim)
                                      for _ in range(self.num_agents)])
        self.v_rsa = torch.nn.TransformerEncoderLayer(
            d_model=self.transformer_dim, nhead=self.num_heads, batch_first=True,
            dim_feedforward=self.transformer_dim, dropout=0)

        self.v_d1 = torch.nn.Linear(self.num_agents * self.transformer_dim, self.embadding_dim)
        #  self.v_d2 = torch.nn.Linear(self.embadding_dim, self.embadding_dim)
        self.v = torch.nn.Linear(self.embadding_dim, 1)

        self.f = torch.nn.ModuleList([torch.nn.Linear(self.state_size + self.obs_size + self.action_size, self.transformer_dim)
                                      for _ in range(self.num_agents)])
        self.q_rsa = torch.nn.TransformerEncoderLayer(
            d_model=2*self.transformer_dim, nhead=self.num_heads, batch_first=True,
            dim_feedforward=self.transformer_dim, dropout=0)

        self.q_d1 = torch.nn.Linear(self.num_agents * 2*self.transformer_dim, self.embadding_dim)
        #self.q_d2 = torch.nn.Linear(self.embadding_dim, self.embadding_dim)
        self.q = torch.nn.Linear(self.embadding_dim, 1)

    def forward(self, states, obs): # t : sequence_index
        b = states.shape[0]
        # (32,3,114)
        states = [s.reshape(b, self.state_size) for s in torch.split(states, 1, dim=1)]
        obs = [o.reshape(b, self.obs_size) for o in torch.split(obs, 1, dim=1)]
        s_embed = [g(torch.cat((s,o), dim=1)) for g, s, o in zip(self.g, states, obs)]
        v_h = self.v_rsa(torch.stack(s_embed, dim=1))
        v_h = F.relu(self.v_d1(v_h.reshape(b, -1)))
        #v_h = F.relu(self.v_d2(v_h))
        v = self.v(v_h)

        return v
    
    def compute_q(self, states, obs, actions, agent_idx):
        b = states.shape[0]
        
        # 원핫 인코딩 생성 (수정)
        onehot_actions = F.one_hot(actions.long(), num_classes=self.action_size).reshape(b, self.num_agents, self.action_size)
        active_actions = (actions != 0) # action = 0은 default action으로, 에이전트가 살아있는 상태에서는 선택할 수 없는 액션임
        onehot_actions *= active_actions # 에이전트가 선택할 수 있는 액션에 대해서만 학습 필요
        
        onehot_actions = torch.split(onehot_actions, 1, dim=1)

        states = [s.reshape(b, self.state_size) for s in torch.split(states, 1, dim=1)]
        obs = [o.reshape(b, self.obs_size) for o in torch.split(obs, 1, dim=1)]
        s_embed = [g(torch.cat((s,o), dim=1)) if i == agent_idx else torch.zeros((b, self.transformer_dim)).to(self.device)
                   for i, (g, s, o) in enumerate(zip(self.g, states, obs))]
        
        
        sa_embed = [torch.zeros((b, self.transformer_dim)).to(self.device) if i == agent_idx else \
                    f(torch.cat((s,o,a.reshape(b, self.action_size)), dim=1))
                    for i, (f, s, o, a) in enumerate(zip(self.f, states, obs, onehot_actions))]
        q_h = self.q_rsa(torch.cat((torch.stack(s_embed, dim=1), torch.stack(sa_embed, dim=1)), dim=2))
        q_h = F.relu(self.q_d1(q_h.reshape(b, -1)))
        #q_h = F.relu(self.q_d2(q_h))
        q = self.q(q_h)

        del states, onehot_actions, active_actions, s_embed, sa_embed
        torch.cuda.empty_cache()

        return q