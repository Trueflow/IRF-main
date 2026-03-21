from modules.Actor.RNNActor import RNNActor
def teamInfo(env, config):
    # team0, team1 알고리즘은 MARL or HRL / LIIR or POCA 식으로 들어옴
    behavior_keys = list(env.behavior_specs.keys())
    print(behavior_keys)
    if config[1].Framework =="rsa":
        team_behavior, team_info =  {0:[]}, {0:[]}
        team_behavior[0].append(behavior_keys[0])
        team_num=1
    else:
        team_behavior, team_info = {0:[],1:[]}, {0:[],1:[]}
        for name in behavior_keys:
            if "team=0" in name:
                team_behavior[0].append(name)
            elif "team=1" in name:
                team_behavior[1].append(name)
        team_num=2
    
    for team in range(0,team_num): # team 0,1에 대해
        
        frame, algorithm = config[team].Framework, config[team].Algorithm
        team_info[team].append(frame) # MARL or HRL 명시
        args = config[team]
        team_info[team].append(args) # hyperparameter
        
        if frame=="marl":
            agent_info = create_base_agent_info(team_behavior[team][0], algorithm, args)
            if algorithm=="poca_single" or algorithm=="poca":
                agent_info["memory"]["next_state"] = []
            team_info[team].append(agent_info)
        elif frame=="hrl":
            upper_info, lower_info = None, None
            for key in team_behavior[team]:
                if "Upper" in key:
                    upper_info = create_upper_agent_info(key, args)
                else:
                    lower_info = create_base_agent_info(key, algorithm, args)
                    # goal 처리를 따로 하지 않았음. goal=목표달성보상=reward(개별)로 따로 들어갈 수 있음.
            team_info[team].extend([upper_info, lower_info])
    
    return team_info

def create_base_agent_info(behavior, algorithm, args):
    """MARL 에이전트와 HRL Lower 에이전트를 위한 기본 정보 생성"""
    return {
        "behavior": behavior,
        "algorithm": algorithm,
        "args": args,
        "agents_id": [],
        "memory": {
            "state": [],
            "obs": [],
            "actions": [],
            "activeSelf": [],
            "reward": [],
            "groupReward": [],
            "done": False,
        },
        "last_action":[] # for cds, emc
    }

def create_upper_agent_info(key, args):
    """HRL Upper 에이전트 정보 생성"""
    return {
        "behavior": key,
        "algorithm": 'ppo',  # 기본은 PPO로 할 것   
        "args": args,
        "upper_id": [],
        "actor": RNNActor(args, args.upper_state, args.upper_action),
        "hidden_state": [RNNActor(args, args.upper_state, args.upper_action).init_hidden_2() 
                         for _ in range(0, args.num_agents)],
        "memory": {
            "obs": [],
            "UpperAction": [],
            "actionMask": [],
            "groupReward": [],
            "reward": [],
            "done": False,
        },
    }

# 알고리즘별 스키마 생성 함수들
def create_liir_scheme(args, team):
    """LIIR 알고리즘용 스키마 생성"""
    episode_info = {
        "scores": [], 
        "r_in_list": [],
        "actor_losses": [], 
        "intrinsic_losses": [], 
        "critic_losses": [], 
        "episode_length": []
    }
    return {"EpisodeInfo": episode_info}

def create_coma_scheme(args, team):
    """COMA 알고리즘용 스키마 생성"""
    episode_info = {
        "scores": [],
        "actor_losses": [],
        "critic_losses": [], 
        "episode_length": []
    }
    return {"EpisodeInfo": episode_info}

def create_poca_scheme(args, team):
    """POCA 알고리즘용 스키마 생성"""
    episode_info = {
        "scores": [],
        "actor_losses": [],
        "critic_losses": [], 
        "episode_length": []
    }
    return {"EpisodeInfo": episode_info}

def create_cds_scheme(args, team):
    episode_info = {
        "scores": [],
        "losses":[],
        "td_error_abs":[],
        "hit_prob":[],
        "intrinsic_rewards":[],
        "episode_length": []
    }
    return {"EpisodeInfo": episode_info}

def create_emc_scheme(args, team):
    episode_info = {
        "scores": [],
        "losses":[],
        "vdn_losses":[],
        "td_error_abs":[],
        "hit_prob":[],
        "intrinsic_rewards":[],
        "episode_length": []
    }
    return {"EpisodeInfo": episode_info}    
    
def create_ppo_scheme(args, team):
    """PPO 알고리즘용 스키마 생성"""
    episode_info = {
        "actor_losses": [], 
        "critic_losses": [],
        "scores": [],
        "episode_length": []
    }
    return {"EpisodeInfo": episode_info}

SCHEMA_REGISTRY = {}
# 레지스트리에 함수 등록
SCHEMA_REGISTRY["liir"] = create_liir_scheme
SCHEMA_REGISTRY["coma"] = create_coma_scheme
SCHEMA_REGISTRY["poca"] = create_poca_scheme
SCHEMA_REGISTRY["cds"] = create_cds_scheme
SCHEMA_REGISTRY["emc"] = create_emc_scheme
SCHEMA_REGISTRY["ppo"] = create_ppo_scheme
    
def WriteSchemeInfo(algorithm, args, team):
    write_scheme = { "score": [], "episode": 0, "done": False, "team": team }
    # 알고리즘별 스키마 생성 및 병합
    if algorithm in SCHEMA_REGISTRY:
        # 레지스트리에서 알고리즘별 스키마 생성 함수 호출
        algorithm_scheme = SCHEMA_REGISTRY[algorithm](args, team)
        # 두 스키마 병합
        write_scheme.update(algorithm_scheme)
    else:
        print(f"Unknown algorithm: {algorithm}")
    
    return write_scheme
