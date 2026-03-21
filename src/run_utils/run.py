from run_utils.mlagents_trans import ( 
    process_obs, process_actionmask, process_active, set_action, 
    process_rewards, process_next_state, process_done, process_state)

from modules.action_selector import action_selector_registry as action_selector
import run_utils.upper as upper

import numpy as np
import torch.cuda
# import gc
# from memory_profiler import profile
        
# @profile
def agents_run(env, Info):
    Agent = Info[-1] # MARL or Lower Agent
    dec, _ = env.get_steps(Agent["behavior"])
    args = Info[1]
    Agent["memory"]["state"] = process_state(dec.obs, args) 
    Agent["memory"]["obs"] = process_obs(dec.obs, args)
    Agent["memory"]["ActionMask"] = process_actionmask(dec.obs, args)
    Agent["memory"]["activeSelf"] = process_active(dec.obs, args)
    # 미리 설정된 액션 선택기 사용
    Agent["memory"]["actions"] = action_selector[Agent["algorithm"]](Agent, args, args.train_mode)
    
    set_action(Agent["memory"]["actions"], env, Agent["behavior"])

    del Agent, dec, _, args
    torch.cuda.empty_cache()
# @profile
def agents_step(env, Info):
    # Lower, Normal Agent가 env.step() 이후 수행하는 메소드
    envType, args, Agent = Info[0], Info[1], Info[-1]

    dec, term = env.get_steps(Agent["behavior"])
    if args.use_next_state:
        next_states, next_obs, next_active_agents = process_next_state(dec, term, args)
    # hrl 구현 시에는 term.agent_id 길이가 에이전트 수와 같은지 확인해야함
    Agent["memory"]["done"] = process_done(term)
    StepReward, Agent["memory"]["reward"] = process_rewards(dec, term, Agent["memory"]["done"])
    # Agent["memory"]["activeSelf"] = env_util.process_active(dec.obs, args)

    Agent["WriteScheme"]["score"].append(StepReward)

    # if envType=='hrl':
    #     upper.upper_step(env, args, [Agent["memory"]["reward"]], Info)

    if args.train_mode:
        _obs, _actions, _actionmask, _active_agents = sample_memory(Agent, args, envType)
        if args.use_next_state:
            _next_state, _next_obs = poca_sample(next_states, next_obs, next_active_agents, args)
            Agent["agent"].append_sample(Agent["memory"]["state"], _obs, _actions, _actionmask, [Agent["memory"]["reward"]],_next_state, _next_obs, [Agent["memory"]["done"]], _active_agents)
            del _next_state, _next_obs, next_states, next_obs, next_active_agents
        else:
            Agent["agent"].append_sample(Agent["memory"]["state"], _obs, _actions, _actionmask, [Agent["memory"]["reward"]], [Agent["memory"]["done"]], _active_agents)

        del _obs, _actions, _actionmask, _active_agents
    del dec, term, envType, args, Agent
    torch.cuda.empty_cache()     

def agents_write(Info):
    # print("agents_write 시작")
    Agent, args = Info[-1], Info[1]
    # print(f"Agent 및 args 가져오기 완료")
    
    # print("train_model 호출 중...")
    try:
        learn_scheme = Agent["agent"].train_model()
        print(f"train_model 완료")
    except Exception as e:
        print(f"train_model에서 오류 발생: {e}")
        print(f"오류 타입: {type(e)}")
        import traceback
        traceback.print_exc()
        raise e
    
    # print("WriteScheme 접근 중...")
    # print(f"Agent keys: {list(Agent.keys())}")
    # print(f"WriteScheme keys: {list(Agent['WriteScheme'].keys())}")
    
    try:
        scheme = Agent["WriteScheme"]["EpisodeInfo"]
        # print(f"EpisodeInfo 접근 성공, scheme keys: {list(scheme.keys())}")
    except Exception as e:
        print(f"EpisodeInfo 접근 실패: {e}")
        print(f"WriteScheme 내용: {Agent['WriteScheme']}")
        raise e

    # print("write_scheme 호출 중...")
    Agent["agent"].write_scheme(scheme, learn_scheme, args)
    # print("write_scheme 완료")
    
    if Info[0]=="hrl":
        Upper = Info[-2]
        upper_learn_scheme = Upper["agent"].train_model()
        Upper["agent"].write_scheme(Upper["WriteScheme"], upper_learn_scheme)
        del Upper, upper_learn_scheme

    # print("메모리 정리 중...")
    del Agent, args, learn_scheme, scheme
    # print("메모리 정리 완료")
    
    import torch
    if torch.cuda.is_available():
        print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        torch.cuda.empty_cache()
        print("GPU 캐시 정리 완료")
    
    # print("agents_write 완료")

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

def poca_sample(next_states, next_obs, next_active_agents, args):
    next_state = np.zeros((args.num_agents, args.state_size))
    next_obs = np.zeros((args.num_agents, args.obs_size))
    for i in range(0,args.num_agents):
        if next_active_agents[i].item()==1.0:
            next_state[i] = next_states[i]
            next_obs[i] = next_obs[i]
    return next_state, next_obs