import torch
import torch.nn.functional as F
import gc
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