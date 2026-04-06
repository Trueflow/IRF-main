from modules.Actor.RNNActor import RNNActor
def teamInfo(env, config):
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
    
    return team_info

def create_base_agent_info(behavior, algorithm, args):
    """MARL 에이전트를 위한 기본 정보 생성"""
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

# 알고리즘별 스키마 생성 함수들
def create_irf_scheme(args, team):
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
    episode_info = {
        "scores": [],
        "actor_losses": [],
        "critic_losses": [], 
        "episode_length": []
    }
    return {"EpisodeInfo": episode_info}

def create_poca_scheme(args, team):
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

SCHEMA_REGISTRY = {}
# 레지스트리에 함수 등록
SCHEMA_REGISTRY["irf"] = create_irf_scheme
SCHEMA_REGISTRY["coma"] = create_coma_scheme
SCHEMA_REGISTRY["poca"] = create_poca_scheme
SCHEMA_REGISTRY["cds"] = create_cds_scheme
SCHEMA_REGISTRY["emc"] = create_emc_scheme
    
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
