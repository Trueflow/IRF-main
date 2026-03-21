import numpy as np
from mlagents_envs.base_env import ActionTuple

def process_state(obs, args):
    maskidx = args.vec_obs + args.action_size
    return obs[args.VEC_IDX][:,maskidx:-2]

def process_obs(obs, args):
    ray_obs = obs[args.RAY_IDX]
    vec_obs = obs[args.VEC_IDX][:,:args.vec_obs]
    # print(f"ray_obs: {ray_obs.shape}, vec_obs: {vec_obs.shape}")
    return np.concatenate((ray_obs, vec_obs), axis=-1)

def process_actionmask(obs, args):
    maskidx = args.vec_obs + args.action_size
    return (obs[args.VEC_IDX][:,args.vec_obs:maskidx].astype(int)).tolist()

def process_active(obs, args):
    return list(obs[args.VEC_IDX][:,-1].astype(int))

def process_done(term):
    return True if len(term.agent_id)!=0 else False

def set_action(actions, env, behavior):
    action_tuple = ActionTuple()
    action_tuple.add_discrete(actions)
    env.set_actions(behavior, action_tuple)

def process_rewards(dec, term, done):
    if not done:
        # reward = dec.reward + dec.group_reward 개인보상 포함이면 list로 나가자
        reward = dec.group_reward
        return np.mean(reward), np.mean(reward)
    else:
        # reward = term.reward + term.group_reward
        reward = term.group_reward
        return np.mean(reward), np.mean(reward)
  
def process_next_state(dec, term, args): # for MA-POCA
    if len(term.agent_id)!=0:
        obs = term.obs
    else:
        obs = dec.obs
    return process_state(obs, args), process_obs(obs, args), process_active(obs, args)