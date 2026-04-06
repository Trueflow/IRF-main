import torch
import torch.nn.functional as F
import numpy as np

def single_actor_selector(AgentInfo, args, training=True):
    actions = []
    actionmask = torch.FloatTensor(AgentInfo["memory"]["ActionMask"]).to(args.device)
    if args.use_last_action:
        obs = torch.FloatTensor(AgentInfo["memory"]["obs"]).to(args.device)
        if len(AgentInfo["last_action"]) != 0: last_action = AgentInfo["last_action"] 
        else: last_action = [0 for _ in range(args.num_agents)]
        last_action_tensor = torch.LongTensor(last_action).to(args.device).reshape(args.num_agents, 1)  # (num_agents,)
        last_action_onehot = F.one_hot(last_action_tensor, num_classes=args.action_size).squeeze(1)  # (num_agents, action_size)
        input_pi = torch.cat([obs, last_action_onehot], dim=-1).unsqueeze(0)  # (1, num_agents, obs_size + action_size)
        hidden = AgentInfo["hidden_state"] 
    else:
        hidden = AgentInfo["hidden_state"] 
        input_pi = torch.FloatTensor(AgentInfo["memory"]["obs"]).to(args.device).unsqueeze(0)
    pi, AgentInfo["hidden_state"] = AgentInfo["actor"].inference_forward(input_pi, hidden)
    pi = pi.squeeze()
    
    if training: # action selection with epsilon-greedy exploration
        epsilon_action_num, epsilon = pi.size(-1), AgentInfo["epsilon_schedule"].epsilon
        pi = (1-epsilon)*pi + actionmask*(epsilon/epsilon_action_num)
    
    masked_pi = pi * actionmask
    for agent in range(args.num_agents):
        action = torch.multinomial(masked_pi[agent], num_samples=1).item() # select action
        actions.append(action)
    
    action = np.array(actions, dtype=object).reshape(args.num_agents, 1)
    AgentInfo["last_action"] = actions
    return action
    
# for multi-actors algorithm
def multi_actor_selector(AgentInfo, args, training=True):
    actions = []
    hidden_states = AgentInfo["hidden_state"]  # hidden states - list form
    obs = torch.FloatTensor(AgentInfo["memory"]["obs"]).to(args.device)
    actionmask = torch.FloatTensor(AgentInfo["memory"]["ActionMask"]).to(args.device)
    
    for agent_idx in range(args.num_agents):
        agent_obs = obs[agent_idx:agent_idx+1].unsqueeze(0)  # [1, 1, state_size]
        agent_mask = actionmask[agent_idx]
        
        agent_hidden = hidden_states[agent_idx] # [1, 1, embadding_dim]

        pi, new_hidden = AgentInfo["actor"][agent_idx].inference_forward(agent_obs, agent_hidden)
        pi = pi.squeeze()

        AgentInfo["hidden_state"][agent_idx] = new_hidden # update hidden states
        
        if training: # action selection with epsilon greedy exploration
            epsilon_action_num, epsilon = pi.size(-1), AgentInfo["epsilon_schedule"].epsilon
            pi = (1-epsilon)*pi + agent_mask*(epsilon/epsilon_action_num)
        
        masked_pi = pi * agent_mask
        action = torch.multinomial(masked_pi, num_samples=1).item() # select action
        actions.append(action)
    
    action = np.array(actions, dtype=object).reshape(args.num_agents, 1)
    return action