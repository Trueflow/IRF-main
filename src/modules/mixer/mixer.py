# CDS_dmaq_qatten.py 
#(https://github.com/lich14/CDS/blob/main/CDS_SMAC/QPLEX-master-SC2/pymarl-master/src/modules/mixers/dmaq_qatten.py)
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
from .qatten_weight import Qatten_Weight
from .si_weight import SI_Weight

class dmaq_Mixer(nn.Module):
    def __init__(self, args):
        super(dmaq_Mixer, self).__init__()

        self.args = args
        self.n_agents = args.num_agents
        self.n_actions = args.action_size
        self.state_dim = int(np.prod(args.state_size))
        self.action_dim = args.num_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1

        self.attention_weight = Qatten_Weight(args)
        self.si_weight = SI_Weight(args)

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = th.sum(agent_qs, dim=-1)
        return v_tot

    def calc_adv(self, agent_qs, states, actions, max_q_i):
        states = states.reshape(-1, self.n_agents, self.state_dim)  # (bs, n_agents, state_dim)
        states = states[:,0,:]
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        adv_w_final = self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)
        
        adv_tot = th.sum(adv_q * (adv_w_final - 1.), dim=1)
        return adv_tot

    def calc(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_q_i)
            return adv_tot

    def forward(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        bs = agent_qs.size(0)
        w_final, v, attend_mag_regs, head_entropies = self.attention_weight(agent_qs, states, actions)
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = v.view(-1, 1).repeat(1, self.n_agents)
        v /= self.n_agents

        agent_qs = agent_qs.view(-1, self.n_agents)
        
        agent_qs = w_final * agent_qs + v
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)
            max_q_i = w_final * max_q_i + v # 여기서 차원 문제제
        # 일단 여기까지 왔어
        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        return v_tot, attend_mag_regs, head_entropies