from copy import deepcopy
from box import Box
import yaml
from torch import device, cuda
import os

def get_config():
    config_dir = 'RLsetting'
    env_config_dir = 'UnityEnv/EnvInfo'

    with open(f"{config_dir}.yaml", "r", encoding="utf-8") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, f"{config_dir}.yaml error: {exc}"

    with open(f"{env_config_dir}.yaml", "r", encoding="utf-8") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, f"{env_config_dir}.yaml error: {exc}"
        env_config = config_dict
    
    team0_config, team1_config, RLcontrol, merged_config = config_env_merge(config, env_config)
    return [Box(team0_config), Box(team1_config)], Box(RLcontrol), merged_config

def config_env_merge(config, unityenv):
    RLcontrol = config["LearningControl"]
    if not RLcontrol["training"]: RLcontrol["run_step"] = 1

    team0, team1 = config["Framework"][0].upper(), config["Framework"][1].upper()
    scenario, game = f"{team0}vs{team1}", RLcontrol["env"]

    if scenario not in unityenv: 
        raise KeyError(f"EnvInfo.yaml에 '{scenario}'시나리오가 없습니다.")
    elif game not in unityenv[scenario]:
        raise KeyError(f"{scenario}에 '{game}' 환경이 없습니다.")
    
    env_config = unityenv[scenario][game]
    merged = {**config, **env_config}
    del merged["LearningControl"]

    optim_name = merged.get("optimiser") 
    param = merged.get("optimiser_param", {}).get(optim_name)
    if param is None: 
        raise KeyError(f"optimiser_param에서 '{optim_name}' 항목을 찾을 수 없습니다.")
    merged["optimiser_param"] = param
    merged["device"] = device("cuda" if cuda.is_available() else "cpu")
    env_info = f"EnvInfo:{game}"
    merged[env_info] = env_config

    team0_config, team1_config = deepcopy(merged), deepcopy(merged)
    for key, values in merged.items():
        if isinstance(values, (list, tuple)) and len(values)==2:
            team0_config[key] = values[0]
            team1_config[key] = values[1]
    team0_cfg = config_alg_merge(team0_config)
    team1_cfg = config_alg_merge(team1_config)

    return team0_cfg, team1_cfg, RLcontrol, merged

def config_alg_merge(config):
    if config["Framework"].lower() != "rsa":
        alg_name = config["Algorithm"]
        with open(f"Config/{alg_name}.yaml", "r", encoding="utf-8") as f:
            try:
                alg_config = yaml.load(f, Loader=yaml.FullLoader)
                final_config = {**alg_config, **config}
                return final_config
                # Algorithm 고유 Configuration 파일과 merge
            except yaml.YAMLError as exc:
                assert False, f"{algconfig_dir}.yaml error: {exc}"
    else: # RSA라면 그냥 Return
        return config

def save_config(config, save_path, filename="hyperparameters.yaml"):
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, filename)
    cfg = config.copy()
    
    # device 객체는 직렬화가 불가능하므로 문자열로 변환
    if "device" in cfg:  # 원본 수정 방지를 위해 복사
        cfg["device"] = str(cfg["device"])

    for alg in range(0,2): # 2번 반복
        if cfg["Framework"][alg].lower() !="rsa":
            alg_name = cfg["Algorithm"][alg]
            with open(f"Config/{alg_name}.yaml", "r", encoding="utf-8") as f:
                alg_config = yaml.load(f, Loader=yaml.FullLoader)
                cfg[f"config_{alg_name}"] = alg_config
    
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    
    # print(f"하이퍼파라미터 설정을 {file_path}에 저장했습니다.")
