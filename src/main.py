from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import EnvSetting.TeamInfo as teaminfo
from EnvSetting.GameInfo import GameSetting

import run_utils.EnvManager as EnvManager
from run_utils.run import agents_run, agents_step, agents_write
from Utils.GetConfig import get_config, save_config
from Utils.parse import parse_args

import torch
import gc
import sys
import traceback

# from pympler import muppy, summary

if __name__ == '__main__':
    print("프로그램 시작")
    env = None
    try:
        arg = parse_args()
        workerId, graphic = arg.workerid, arg.graphic
        # no_graphic_setting = False  # True if graphic=="False" else False
        graphic_option = True if graphic=="False" else False
        configlist, ENVargs, merged_config = get_config() # config 파일 불러오기 [team0, team1] 순서
        env_name, save_path, load_path = GameSetting(configlist, ENVargs.env) # 유니티 환경 경로 설정 (file_name)
        save_config(merged_config, save_path) # 현재 하이퍼파라미터 설정을 파일로 저장
        
        # 유니티 환경 초기화
        engine_configuration_channel = EngineConfigurationChannel()
        env = UnityEnvironment(file_name=env_name, side_channels=[engine_configuration_channel],worker_id=workerId,no_graphics=graphic_option)
        env.reset()
        engine_configuration_channel.set_configuration_parameters(time_scale=20.0, quality_level=1)
        # 원래 학습시 Time Scale 20으로 설정 (영상용 : 2)
        # 설정한 환경 출력
        print(f'TrainStep : {ENVargs.run_step} | TestStep: {ENVargs.test_step}\n')
        TeamInfo = teaminfo.teamInfo(env, configlist)
        RSAmode = True if configlist[1].Framework=="rsa" else False
        startStep, win_count = 0, {0:0, 1:0} # team0 / team1 이긴 횟수

        for team, info in TeamInfo.items(): 
            EnvManager.InitialSetting(team, info, env, save_path, load_path[team])
        for step in range(1, ENVargs.run_step+ENVargs.test_step+1):
            if step == ENVargs.run_step: # 학습이 끝났는지 확인
                EnvManager.LearningEnd(step, ENVargs, TeamInfo)
                env.reset()
                engine_configuration_channel.set_configuration_parameters(time_scale=20.0)
                win_count = {0:0, 1:0}
            for team, Info in TeamInfo.items(): agents_run(env, Info)

            env.step() # Proceed to next step

            for team, Info in TeamInfo.items(): agents_step(env, Info)

            if TeamInfo[0][-1]["memory"]["done"]:
                if (step-startStep)>99:
                    EnvManager.calculate_win(env, TeamInfo, win_count, RSAmode)
                    if ENVargs.training:
                        for team, Info in TeamInfo.items(): 
                            if Info[1].train_mode: 
                                agents_write(Info) # learning method
                        print(f"step {step}, episode end - Learning End")
                    EnvManager.episodeEnd(step, startStep, TeamInfo, ENVargs, win_count)
                    gc.collect()
                    torch.cuda.empty_cache()
                EnvManager.memoryClear(TeamInfo)
                startStep = step+1
            # step Memory 정리
            for team, Info in TeamInfo.items():
                Info[-1]["memory"].clear()
            torch.cuda.empty_cache()

    except RuntimeError as e:
        # CUDA Out of Memory 또는 기타 런타임 오류 처리
        traceback.print_exc()
        sys.exit(1)  # 오류 코드 1로 종료
    except Exception as e:
        # 기타 예외 처리
        traceback.print_exc()
        sys.exit(1)  # 오류 코드 1로 종료
    finally:
        # Unity 환경 종료
        if env is not None:
            env.close()
            print("Unity 환경이 안전하게 종료되었습니다.")