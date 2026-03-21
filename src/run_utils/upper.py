import torch
from modules.action_selector.upper_actor import upper_action_selector
from run_utils.mlagents_trans import set_action, process_done
import numpy as np

def upper_run(env, args, Info):
    Upper, Lower = Info[-2], Info[-1]
    UpperInfo = Upper[-2]
    dec, _ = env.get_steps(UpperInfo["behavior"])

    UpperInfo["obs"] = dec.obs[0] # vector obs 만 존재
    UpperInfo["actionMask"] = np.array(dec[UpperInfo["upper_id"]].action_mask, dtype=bool)
    UpperInfo["UpperAction"] = upper_action_selector(UpperInfo, UpperInfo["actionMask"], args, args.train_mode) # upperAgent가 선택한 옵션 (function으로 만들어야함) 
    set_action(UpperInfo["UpperAction"], env, UpperInfo["behavior"])
    UpperInfo["actionMask"] = UpperInfo["actionMask"].astype(float)

    del dec, _, UpperInfo, 

def upper_step(env, args, groupreward, Info):
    # upper Agent가 env.step() 이후 수행하는 메소드
    # upper의 특성에 맞게 구성
    Upper = Info[-2]
    dec, term = env.get_steps(Upper["behavior"])
    Upper["done"] = process_done(term)
    Upper["reward"] = process_upper_reward(dec, term, Upper["done"])
    Upper["groupReward"] = np.mean(groupreward)
    if args.train_mode:
        Upper["agent"].append_sample(np.array(Upper["obs"], dtype=float), Upper["UpperAction"].astype(int), [Upper["groupReward"]], Upper["reward"], [Upper["done"]], Upper["actionMask"])
    del Upper, groupreward, dec, term

def process_upper_reward(dec, term, done):
    return dec.reward if not done else term.reward