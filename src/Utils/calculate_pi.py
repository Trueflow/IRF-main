import torch
import torch.nn.functional as F
import gc
# from memory_profiler import profile
import tracemalloc

def decentralized_pi(policy, state, action, actionmask, active, args, eps, training=True):
    policy.flattenParameters()
    hidden_state = policy.init_hidden()
    pi, _ = policy.forward(state, hidden_state) # dhstate : (sq_length, value)
    if training: # Epsilon-greedy exploration 적용
        epsilon_action_num = pi.size(-1)
        pi = (1-eps)*pi + actionmask*(eps/epsilon_action_num)
    masked_pi = pi * actionmask
    
    sum_masked = masked_pi.sum(dim=-1, keepdim=True)
    masked_pi = torch.where(sum_masked == 0, torch.ones_like(sum_masked), masked_pi)  # 분모가 0이면 uniform 값 사용
    pi = masked_pi / masked_pi.sum(dim=-1, keepdim=True)
    pi_taken = torch.gather(pi, dim=-1, index=action.long())

    # pi_active = pi_taken * active + (1-active) # active=0 즉,비활성화 상태일떄 학습에 영향 X (->pi_active = 1 -> log_pi_taken = 0)
    log_pi_taken = torch.log(pi_taken.squeeze(-1) + 1e-9)
    del pi, hidden_state, pi_taken, masked_pi, sum_masked
    torch.cuda.empty_cache()
    return log_pi_taken


def decentralized_pi_ppo(policy, state, action, action_mask, args, training=True):
    # for upper agents
    # print(f"state : {state.shape} | action : {action.shape} | actionMask : {action_mask.shape}")
    hidden_state = [policy.init_hidden(state) for _ in range(0, args.num_agents)]
    branch_entropies = []
    for branch in range(0,args.num_agents):
        # 각 branch에 대한 확률 계산
        agent_out, _ = policy(state, hidden_state[branch], False)
        action_mask_ = action_mask[...,branch,:]
        agent_out = agent_out.masked_fill(action_mask_==0, -float('inf'))
        if action_mask_.sum() == 0:
            agent_out = torch.zeros_like(agent_out)
        agent_out = agent_out - torch.max(agent_out, dim=-1, keepdim=True)[0]
        # agent_out[~action_mask[branch]] = float('-inf')
        # agent_out = agent_out / torch.clamp(torch.sum(agent_out, dim=-1, keepdim=True), min=1e-8)
        # print(f"Agent Out min: {agent_out.min()}, max: {agent_out.max()}")
        pi = torch.softmax(agent_out, dim=-1)
        pi = pi / torch.sum(pi, dim=-1, keepdim=True)
        entropy = -torch.sum(pi * torch.log(pi + 1e-8), dim=-1)
        branch_entropies.append(entropy)
        # print(f"pi min: {pi.min()}, max: {pi.max()}")
        # if training: # Epsilon-greedy exploration 적용
            # epsilon_action_num = agent_out.size(-1)
            # pi = (1-args.epsilon)*pi + torch.ones_like(pi)*(args.epsilon/epsilon_action_num)
            # if not torch.isclose(pi.sum(dim=-1), torch.tensor(1.0)).all():
            #    print("Warning: pi is not a valid probability distribution after Epsilon-Greedy")
        pi_branch = torch.gather(pi, dim=-1, index=action[...,branch].long().unsqueeze(0)).squeeze(-1)
        pi_branch = torch.clamp(pi_branch, min=1e-8)
        # if torch.isnan(pi_branch).any() or torch.isinf(pi_branch).any():
        #    print("NaN or Inf detected in pi_branch")
        if branch!=0:
            log_pi_taken += torch.log(pi_branch+1e-8)
        else:
            log_pi_taken = torch.log(pi_branch+1e-8)

    dist_entropy = torch.stack(branch_entropies).mean()
    torch.cuda.empty_cache()
    del agent_out, action_mask_, pi, pi_branch
    return log_pi_taken, dist_entropy
