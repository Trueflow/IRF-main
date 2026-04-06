from run_utils.mlagents_trans import ( 
    process_obs, process_actionmask, process_active, set_action, 
    process_rewards, process_next_state, process_done, process_state)

from modules.action_selector import action_selector_registry as action_selector
import run_utils.upper as upper

import numpy as np
import torch.cuda
import torch

def agents_run(env, Info):
    Agent = Info[-1] # MARL or Lower Agent
    dec, _ = env.get_steps(Agent["behavior"])
    args = Info[1]
    Agent["memory"]["state"] = process_state(dec.obs, args) 
    Agent["memory"]["obs"] = process_obs(dec.obs, args)
    Agent["memory"]["ActionMask"] = process_actionmask(dec.obs, args)
    Agent["memory"]["activeSelf"] = process_active(dec.obs, args)
    # Action Selector is already set in modules/action_selector/init
    Agent["memory"]["actions"] = action_selector[Agent["algorithm"]](Agent, args, args.train_mode)
    
    set_action(Agent["memory"]["actions"], env, Agent["behavior"])

    del Agent, dec, _, args
    torch.cuda.empty_cache()

def agents_step(env, Info):
    envType, args, Agent = Info[0], Info[1], Info[-1]

    dec, term = env.get_steps(Agent["behavior"])
    if args.use_next_state:
        next_states, next_obs, next_active_agents = process_next_state(dec, term, args)
    Agent["memory"]["done"] = process_done(term)
    StepReward, Agent["memory"]["reward"] = process_rewards(dec, term, Agent["memory"]["done"])

    Agent["WriteScheme"]["score"].append(StepReward)

    if args.train_mode:
        _obs, _actions, _actionmask, _active_agents = sample_memory(Agent, args, envType)
        if args.use_next_state:
            _next_state, _next_obs = next_sample(next_states, next_obs, next_active_agents, args)
            Agent["agent"].append_sample(Agent["memory"]["state"], _obs, _actions, _actionmask, [Agent["memory"]["reward"]],_next_state, _next_obs, [Agent["memory"]["done"]], _active_agents)
            del _next_state, _next_obs, next_states, next_obs, next_active_agents
        else:
            Agent["agent"].append_sample(Agent["memory"]["state"], _obs, _actions, _actionmask, [Agent["memory"]["reward"]], [Agent["memory"]["done"]], _active_agents)

        del _obs, _actions, _actionmask, _active_agents
    del dec, term, envType, args, Agent
    torch.cuda.empty_cache()     

def agents_write(Info):
    Agent, args = Info[-1], Info[1]
    try:
        learn_scheme = Agent["agent"].train_model()
        print(f"train_model 완료")
    except Exception as e:
        print(f"train_model에서 오류 발생: {e}")
        print(f"오류 타입: {type(e)}")
        import traceback
        traceback.print_exc()
        raise e
    
    try:
        scheme = Agent["WriteScheme"]["EpisodeInfo"]
    except Exception as e:
        print(f"EpisodeInfo 접근 실패: {e}")
        print(f"WriteScheme 내용: {Agent['WriteScheme']}")
        raise e

    Agent["agent"].write_scheme(scheme, learn_scheme, args)
    del Agent, args, learn_scheme, scheme
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def sample_memory(Agent, args, envtype):
    obs = np.zeros((args.num_agents, args.obs_size)) if envtype=="hrl" else np.zeros((args.num_agents, args.obs_size))
    actionmask = np.zeros((args.num_agents, args.action_size))
    actions = np.zeros((args.num_agents, 1))
    active_agents = np.zeros((args.num_agents, 1))
    for i in range(0,args.num_agents):
            if Agent["memory"]["activeSelf"][i].item()==1.0:
                obs[i] = Agent["memory"]["obs"][i]
                actionmask[i] = Agent["memory"]["ActionMask"][i]
                actions[i] = Agent["memory"]["actions"][i]
                active_agents[i] = 1
    return obs, actions, actionmask, active_agents

def next_sample(next_states, next_obs, next_active_agents, args):
    next_state = np.zeros((args.num_agents, args.state_size))
    next_obs = np.zeros((args.num_agents, args.obs_size))
    for i in range(0,args.num_agents):
        if next_active_agents[i].item()==1.0:
            next_state[i] = next_states[i]
            next_obs[i] = next_obs[i]
    return next_state, next_obs