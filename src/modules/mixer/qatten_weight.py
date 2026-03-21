import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl

# races = { "R": sc_common.Random, "P": sc_common.Protoss, "T": sc_common.Terran,"Z": sc_common.Zerg,}

class Qatten_Weight(nn.Module):

    def __init__(self, args):
        super(Qatten_Weight, self).__init__()

        self.name = 'qatten_weight'
        self.args = args
        self.n_agents = args.num_agents
        self.state_dim = int(np.prod(args.state_size))
        # state: 환경정보 9 + 유닛별 정보 8 (8 * num_agents)
        self.unit_dim = args.unit_dim
        self.unit_state_offset = getattr(args, "unit_state_offset", 0)
        self.n_actions = args.action_size
        self.sa_dim = self.state_dim + self.n_agents * self.n_actions
        self.n_head = args.num_heads  # attention head num (default: 2)

        self.embed_dim = args.mixing_embed_dim
        self.attend_reg_coef = args.attend_reg_coef

        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()

        hypernet_embed = self.args.hypernet_embed
        for i in range(self.n_head):  # multi-head attention
            selector_nn = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed), nn.ReLU(), nn.Linear(hypernet_embed, self.embed_dim, bias=False))
            self.selector_extractors.append(selector_nn)  # query
            self.key_extractors.append(nn.Linear(self.unit_dim, self.embed_dim, bias=False))  # key
        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions):
        # states: (episode_num, seq_length, n_agents, state_dim) → (batch_size, state_dim)로 변환
        # batch_size = episode_num * seq_length
        states = states.reshape(-1, self.n_agents, self.state_dim)  # (bs, n_agents, state_dim)
        states = states[:, 0, :]  # (bs, state_dim) - 모든 에이전트가 같은 state를 받으므로 첫 번째만 사용
        
        # global state에서 모든 에이전트의 unit states 추출 (원본 코드 방식)
        unit_states_start = self.unit_state_offset
        unit_states_end = unit_states_start + self.unit_dim * self.n_agents
        unit_states = states[:, unit_states_start:unit_states_end]  # (bs, unit_dim * n_agents)
        unit_states = unit_states.reshape(-1, self.n_agents, self.unit_dim)  # (bs, n_agents, unit_dim)
        unit_states = unit_states.permute(1, 0, 2)  # (n_agents, bs, unit_dim)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # agent_qs: (batch_size, 1, agent_num)
        # states: (batch_size, state_dim)
        all_head_selectors = [sel_ext(states) for sel_ext in self.selector_extractors]
        # all_head_selectors: (head_num, batch_size, embed_dim)
        # unit_states: (agent_num, batch_size, unit_dim)
        all_head_keys = [[k_ext(enc) for enc in unit_states] for k_ext in self.key_extractors]
        # all_head_keys: (head_num, agent_num, batch_size, embed_dim)

        # calculate attention per head
        head_attend_logits = []
        head_attend_weights = []
        for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):
            # curr_head_keys: (agent_num, batch_size, embed_dim)
            # curr_head_selector: (batch_size, embed_dim)
            # (batch_size, 1, embed_dim) * (batch_size, embed_dim, agent_num)
            attend_logits = th.matmul(curr_head_selector.view(-1, 1, self.embed_dim), th.stack(curr_head_keys).permute(1, 2, 0))
            # attend_logits: (batch_size, 1, agent_num)
            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(self.embed_dim)
            attend_weights = F.softmax(scaled_attend_logits, dim=2)  # (batch_size, 1, agent_num)

            head_attend_logits.append(attend_logits)
            head_attend_weights.append(attend_weights)
        
        head_attend = th.stack(head_attend_weights, dim=1)  # (batch_size, self.n_head, self.n_agents)
        head_attend = head_attend.view(-1, self.n_head, self.n_agents)

        v = self.V(states).view(-1, 1)  # v: (bs, 1)
        # head_qs: [head_num, bs, 1]

        head_attend = th.sum(head_attend, dim=1)

        # regularize magnitude of attention logits
        attend_mag_regs = self.attend_reg_coef * sum((logit**2).mean() for logit in head_attend_logits)
        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1).mean()) for probs in head_attend_weights]
        
        return head_attend, v, attend_mag_regs, head_entropies
