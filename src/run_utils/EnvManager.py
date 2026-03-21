from modules.Actor import REGISTRY as mac_registry
#from modules.Actor.RNNActor import RNNActor
from EnvSetting.TeamInfo import WriteSchemeInfo, SCHEMA_REGISTRY
from Algorithm import REGISTRY as A_Registry
from Utils.epsilon_schedule import epsilon_schedule
from mlagents_envs.environment import UnityEnvironment, ActionTuple

import gc
import torch
import numpy as np

def InitialSetting(team, info, env, save_path, load_path):
    if info[0]=="marl":
            dec, term = env.get_steps(info[-1]["behavior"])
            info[-1]["agents_id"] = [id for id in dec.agent_id]
            info[-1]["algorithm"] = info[-1]["algorithm"].lower()
            algorithm, args = info[-1]["algorithm"], info[-1]["args"]
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # position= dec.obs[args.VEC_OBS][:,-2]
            
            # POCA 알고리즘은 여러 액터를 사용 / (사실 rnn_actor 인자에 action_size는 필요없음..)
            if algorithm=="poca":
                # 각 에이전트마다 별도의 액터 생성. input_size = obs_size
                info[-1]["actor"] = [mac_registry["rnn"](args, args.obs_size, args.action_size) for _ in range(args.num_agents)]
                # 각 에이전트마다 별도의 hidden state 생성 - init_hidden_2 사용
                info[-1]["hidden_state"] = [actor.init_hidden_2() for actor in info[-1]["actor"]]
            if args.use_last_action:
                input_size = args.obs_size+args.action_size
                info[-1]["actor"] = mac_registry["rnn"](args, input_size, args.action_size)
                info[-1]["hidden_state"] = info[-1]["actor"].init_hidden()
            else:
                # 다른 알고리즘은 단일 액터 사용
                info[-1]["actor"] = mac_registry["rnn"](args, args.obs_size, args.action_size)
                info[-1]["hidden_state"] = info[-1]["actor"].init_hidden()
                
            # print(f"알고리즘 이름: '{info[-1]['algorithm']}'")
            # print(f"사용 가능한 스키마: {list(SCHEMA_REGISTRY.keys())}")
            info[-1]["WriteScheme"] = WriteSchemeInfo(info[-1]["algorithm"], info[-1]["args"], team)
            # print(f"WriteScheme 생성 완료: {list(info[-1]['WriteScheme'].keys())}")
            info[-1]["epsilon_schedule"] = epsilon_schedule(args.eps_greedy)
            info[-1]["epsilon_schedule"].init_schedule(args.train_mode)
            
            info[-1]["agent"] =  A_Registry[algorithm](info[-1]["args"], info[-1]["actor"], args.eps_greedy.start, 
                                                       save_path+f"/team{team}MARL-{algorithm}",load_path)
            info[-1]["agent"].SetOptimiser()
            # ray_size, vec_size = dec.obs[args.RAY_OBS].shape, dec.obs[args.VEC_OBS][:,:-1].shape
            # print(f"[Observation] Team0 : ray {ray_size} / vec {vec_size}")
            del dec, term, algorithm, args
            torch.cuda.empty_cache()
    elif info[0]=="hrl": # hrl [frame, upper, lower]
        upperdec, upperterm = env.get_steps(info[-2]["behavior"])
        info[-2]["upper_id"] = upperdec.agent_id[0]
        info[-2]["agent"] = A_Registry[info[-2]["algorithm"]](info[-2]["args"], info[-2]["actor"], save_path+f"/team{team}Upper", load_path+f"/team{team}Upper")
        info[-2]["WriteScheme"] = WriteSchemeInfo(info[-2]["algorithm"], info[-2]["args"], team)

        lowerdec, lowerterm = env.get_steps(info[-1]["behavior"])
        algorithm, args = info[-1]["algorithm"], info[-1]["args"]
        # position= lowerdec.obs[args.VEC_OBS][:,-2]

        info[-1]["agents_id"] = [id for id in lowerdec.agent_id]
        info[-1]["actor"] = [RNNActor(args, args.obs_size, args.action_size) for i in range(0, args.num_agents)]
        info[-1]["hidden_state"] = [RNNActor(args, args.obs_size, args.action_size).init_hidden() for i in range(0,args.num_agents)]
        info[-1]["epsilon_schedule"] = epsilon_schedule(args.eps_greedy)
        info[-1]["epsilon_schedule"].init_schedule(args.train_mode)
        info[-1]["WriteScheme"] = WriteSchemeInfo(info[-1]["algorithm"], info[-1]["args"], team)
        info[-1]["agent"]= A_Registry[algorithm](args, info[-1]["actor"], args.eps_greedy.start, save_path+f"/team{team}Lower-{algorithm}", load_path+f"/team{team}Lower")
        info[-1]["agent"].SetOptimiser(info[1])
        del upperdec, upperterm, lowerdec, lowerterm, algorithm
        torch.cuda.empty_cache()

def episodeEnd(step, startStep, TeamInfo, ENVargs, win_count):
    for team, info in TeamInfo.items():
        args = info[1]
        # if info[0]=="hrl":
        #     upper_scheme = info[-2]["WriteScheme"]
        #     upper_scheme["episode"] +=1
        #     upper_scheme["scores"].append(np.sum(upper_scheme["score"], axis=0))
        scheme = info[-1]["WriteScheme"]
        scheme["EpisodeInfo"]["episode_length"].append(step - startStep)
        scheme["episode"] +=1
        scheme["EpisodeInfo"]["scores"].append(np.sum(scheme["score"], axis=0))
        if args.train_mode:
            info[-1]["agent"].epsilon = info[-1]["epsilon_schedule"].update_epsilon(scheme["episode"])

        # scheme["active_agents"] = info[-1][agents_id]
        if scheme["episode"] % ENVargs.print_interval==0:
            win_rate = win_count[team] / sum(win_count.values()) if sum(win_count.values())>0 else 0
            if ENVargs.training==args.train_mode:
                if info[0]=="hrl":
                    info[-2]["agent"].write_summary(scheme, info[-2]["WriteScheme"], ENVargs, win_rate)
                info[-1]["agent"].write_summary(scheme, step, ENVargs, win_rate)

        if args.train_mode and scheme["episode"] % ENVargs.save_interval==0:
            info[-1]["agent"].save_model() # lower&marl
            if info[0]=="hrl": info[-2]["agent"].save_model() # upper
        del args
    torch.cuda.empty_cache()

def memoryClear(TeamInfo):
    for team, info in TeamInfo.items():
        info[-1]["WriteScheme"]["score"].clear()
        info[-1]["agent"].memoryClear()

def calculate_win(env, TeamInfo, win_count, RSAmode):
    for team, info in TeamInfo.items():
        args = info[1]
        _, term = env.get_steps(info[-1]["behavior"])
        # print(term.obs)
        isWin = term.obs[args.VEC_IDX][0,-2].astype(int)
        if isWin==1: win_count[team] += 1
        if RSAmode:
            if isWin==-1: win_count[1] += 1

def LearningEnd(step, ENVargs, TeamInfo):
    for _, info in TeamInfo.items():
        args = info[1]
        if args.train_mode:
            info[-1]["agent"].save_model()
            # if info[0]=="hrl":
               # info[-2]["agent"].save_model()
            info[-1]["WriteScheme"]["score"].clear()
            info[-1]["WriteScheme"]["episode"] = 0
            info[-1]["agent"].memoryClear()
            args.train_mode = False # team hyperparameter
        
    ENVargs.training = False
    print("Test Start")


