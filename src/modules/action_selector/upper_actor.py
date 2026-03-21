import torch
import numpy as np
import torch.nn.functional as F

def upper_action_selector(UpperInfo, action_masks, args, training=True):
    # id = UpperInfo["upper_id"][0]
    state = UpperInfo["obs"]
    actor = UpperInfo["actor"]
    # actor.train(training)

    # action_masks = decision_steps[id].action_mask
    actions = []
    for branch in range(0, args.num_agents):
        hidden_state = UpperInfo["hidden_state"][branch]
        pi, h = actor(torch.FloatTensor(state).to(args.device).unsqueeze(0), hidden_state)
        action_mask = torch.FloatTensor(action_masks[branch]).to(args.device).reshape(1,1,-1)

        # pi[~action_mask] = float('-inf')
        pi = pi * action_mask
        # pi = pi / torch.sum(pi, dim=-1, keepdim=True)
        pi = pi.squeeze(0)
        action = torch.multinomial(pi.squeeze(-1), num_samples=1).item()
        actions.append(action)
        UpperInfo["hidden_state"][branch]=h

    del actor, hidden_state
    return np.array(actions, dtype=object).reshape(1, args.num_agents)