import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from modules.Critic.LIIRcritic import LIIRcritic
from Utils.calculate_pi import decentralized_pi
from datetime import datetime

# MAPOCAAgent 클래스 -> MAPOCA 알고리즘을 위한 다양한 함수 정의 
class LIIRagent:
    def __init__(self, args, actor, epsilon, save_path, load_path):  
        self.args = args
        self.algorithm = "liir"
        self.n_agents = args.num_agents
        self.state_size = args.state_size
        self.n_actions = args.action_size
        self.grad_norm_clip = args.grad_norm_clip
        self.critic_training_interval = 0
        self.target_update_interval = args.target_update_interval

        self.save_path = save_path
        self.load_path = load_path
        self.device = args.device

        self._lambda = args._lambda
        self.epsilon = epsilon
        # self.mu = args.mu
        self.td_lambda = args.td_lambda
        self.gamma = args.gamma
        self.vt_coef = args.vt_coef

        self.critic = LIIRcritic(args).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)   

        #self.actor = RNNActor(args)
        self.actor = actor
        self.policy_old = copy.deepcopy(self.actor)
        self.policy_new = copy.deepcopy(self.actor)

        #self.rnn_params = list(self.actor.parameters())
        self.actor_param = self.actor.parameters()
        self.critic_params = list(self.critic.fc1.parameters())+list(self.critic.fc2.parameters())+list(self.critic.v_mix.parameters())
        self.intrinsic_params = list(self.critic.r_in.parameters()) + list(self.critic.v_ex.parameters())
        self.params = self.critic_params + self.intrinsic_params
        
        self.memory = list()
        self.writer =SummaryWriter(save_path)
    
    def SetOptimiser(self):
        OptimClass = getattr(torch.optim, self.args.optimiser)
        self.actor_optimiser     = OptimClass(self.actor_param,      **self.args.optimiser_param)
        self.critic_optimiser    = OptimClass(self.critic_params,    **self.args.optimiser_param)
        self.intrinsic_optimiser = OptimClass(self.intrinsic_params, **self.args.optimiser_param)
        
        if self.args.load_model == True:
            checkpoint = torch.load(self.load_path + '/ckpt', map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.actor_optimiser.load_state_dict(checkpoint[f"actor_optimizer"])
            self.actor.flattenParameters()
            self.critic.load_state_dict(checkpoint["critic"])
            self.critic_optimiser.load_state_dict(checkpoint["critic_optimizer"])
            self.intrinsic_optimiser.load_state_dict(checkpoint["intrinsic_optimizer"])
            print(f"... Load Model from {self.load_path}/ckpt complete ...")
    
    # 리플레이 메모리에 데이터 추가 (상태, 행동, 그룹보상, 개인보상, 게임 종료 여부, 에이전트 활성 여부)
    def append_sample(self, states, obs, actions, actionmask, Reward, done, actives):
            self.memory.append((states, obs, actions, actionmask, Reward, done, actives))

    def memoryClear(self):
        self.memory.clear()
             
    # 학습 수행
    # @profile
    def train_model(self):
        self.actor.train()
        self.critic.train()

        # 메모리에서 필요한 인자 받아오기
        states      = np.stack([m[0] for m in self.memory], axis=0)
        obs         = np.stack([m[1] for m in self.memory], axis=0)
        actions     = np.stack([m[2] for m in self.memory], axis=0)
        actionmask  = np.stack([m[3] for m in self.memory], axis=0)
        Reward      = np.stack([m[4] for m in self.memory], axis=0)
        done        = np.stack([m[5] for m in self.memory], axis=0)
        actives     = np.stack([m[6] for m in self.memory], axis=0)

        self.memoryClear()

        states, obs, actions, actionmask, Reward, done, actives = map(lambda x: torch.FloatTensor(x).to(self.device),[states, obs, actions, actionmask, Reward, done, actives])
        
        # 학습 이터레이션 시작
        # print(f"state : {states.shape} \n actions : {actions.shape} \n \
        #       reward : {reward.shape} \n done : {done.shape} \n actives : {actives.shape}")
        torch.autograd.set_detect_anomaly(True)
        
        # 1. critic-loss 계산
        target_ex, q_vals, target_mix, r_in, v_ex = self.train_critic(states, obs, actions, Reward, done)
        # 2. actor-loss 계산 (decentralized)
        actor_loss, log_pi_taken = self.train_actor(obs, actions, actionmask, actives, q_vals, target_mix.clone())
        
        # 3. intrinsic-loss 계산 (pi-old pi-new 비교)
        intrinsic_loss = self.train_intrinsic(obs, actions, actionmask, actives, target_mix, target_ex, log_pi_taken, v_ex)    

        self.intrinsic_optimiser.zero_grad()
        intrinsic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.intrinsic_params, self.grad_norm_clip)
        self.intrinsic_optimiser.step()
        critic_loss = []
        for t in reversed(range(states.size(0))):
            q_t, _, _ = self.critic(states, obs, actions, t)
            targets_t = target_mix[t]
            loss = F.mse_loss(q_t.view(self.n_agents), targets_t.detach())
            self.critic_optimiser.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critic_params, self.grad_norm_clip)
            self.critic_optimiser.step()
            #critic_loss_ = critic_loss.clone().detach().cpu()
            critic_loss.append(loss.item())
        del q_t, targets_t, loss


        self.policy_old.load_state_dict(self.actor.state_dict())

        if (self.critic_training_interval+1) % self.target_update_interval == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())

        # r_in_= torch.mean(r_in.detach(), dim=0)

        learn_scheme = {
            "r_in":r_in.reshape(self.n_agents, -1).detach(),
            "critic_loss":critic_loss,
            "actor_loss":actor_loss,
            "intrinsic_loss":intrinsic_loss.item()
        }
        del target_ex, q_vals, target_mix, log_pi_taken, states, actions, Reward, done, actives, critic_loss, actor_loss, intrinsic_loss
        torch.cuda.empty_cache()
        self.critic_training_interval+=1
        return learn_scheme
    
    # @profile
    def train_critic(self, states, obs, actions, Reward, done):
        target_vals, target_val_v_ex, _ = self.target_critic(states, obs, actions)
        q, v_ex, r_explore = self.critic(states, obs, actions)
        
        r_explore_taken = r_explore.gather(dim=-1, index=actions.long())

        target_vals = target_vals.squeeze(-1)

        r_in = r_explore_taken.squeeze(-1) # + self.mu * reward
        r_combined = self._lambda * r_in + Reward

        
        ret = torch.zeros(target_vals.shape[0]+1, target_vals.shape[1]).to(self.device)
        ret_ex = torch.zeros(target_val_v_ex.shape[0]+1, target_val_v_ex.shape[1]).to(self.device)
        ret[-1] = target_vals[-1]*(1 - torch.sum(done, dim=0)) # 마지막 값만 설정
        ret_ex[-1]= target_val_v_ex[-1]*(1 - torch.sum(done, dim=0)) # 마지막 값만 설정
        # Backwards  recursive  update  of the "forward  view"
        with torch.no_grad():
            for t in reversed(range(ret.shape[0]-2)):  
                # 1. ret : targets_mix(= coma targets)
                ret[t] = self.td_lambda*self.gamma*ret[t+1] + r_combined[t] + (1-self.td_lambda)*self.gamma*target_vals[t]*(1-done[t])
                # 2. ret_ex : targets_ex
                ret_ex[t] = self.td_lambda*self.gamma*ret_ex[t+1] + Reward[t] + (1-self.td_lambda)*self.gamma*target_val_v_ex[t]*(1 - done[t])
        target_mix, target_ex = ret[:-1], ret_ex[:-1]
        q_vals, critic_losses = torch.zeros_like(target_mix), torch.zeros(states.size(0)).to(self.device)
        critic_losses = torch.zeros(states.size(0)).to(self.device)
        with torch.no_grad():
            for t in reversed(range(states.size(0))):
                q_t, _, _ = self.critic(states, obs, actions, t)
                q_vals[t] = q_t.view(self.n_agents) # 3
        del ret, ret_ex, r_explore, target_vals, target_val_v_ex, r_combined
        torch.cuda.empty_cache()
        return target_ex, q_vals.detach(), target_mix, r_in, v_ex
    
    def train_actor(self, obs, actions, actionmask, actives, q_vals, target_mix):
        log_pi_taken = decentralized_pi(self.actor, obs, actions, actionmask, actives, self.args, self.epsilon)
        log_pi_taken = log_pi_taken * actives.squeeze(-1)
        advantage = (target_mix.detach() - q_vals)
        adv_std = advantage.std()
        if adv_std<1e-12 or torch.isnan(adv_std):
            advantage = advantage - advantage.mean()
        else:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        # 원본에서 변경된 부분: 살아있는 에이전트만 기준 - 실제로 계산된 loss만 기준으로 함)    
        actor_loss = - (advantage * log_pi_taken.squeeze(-1)).sum() / actives.sum()
        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_param, self.grad_norm_clip)
        self.actor_optimiser.step()
        actor_loss_ = actor_loss.item()
        del advantage, target_mix, actor_loss
        torch.cuda.empty_cache()
        return actor_loss_, log_pi_taken.clone()
    
    # @profile
    def train_intrinsic(self, obs, actions, actionmask, actives, target_mix, target_ex, log_pi_taken, v_ex):
        # 3.1 v_ex 관련 (v_ex_loss, adv_ex)
        v_ex_loss = F.mse_loss(v_ex, target_ex)
        adv_ex = (target_ex - v_ex.clone().detach()).detach()
        std_adv_ex = adv_ex.std()
        if std_adv_ex < 1e-12 or torch.isnan(std_adv_ex):
            adv_ex = adv_ex - adv_ex.mean()
        else:
            adv_ex = (adv_ex - adv_ex.mean()) / (adv_ex.std() + 1e-8)
        # adv_ex = adv_ex.squeeze(0)

        log_pi_taken_old = decentralized_pi(self.policy_old, obs, actions, actionmask, actives, self.args, self.epsilon, training=False)
        log_pi_taken_old = log_pi_taken_old * actives.squeeze(-1)
        # 3.2.2. pg2 : new pi theta (policy_new)
        with torch.no_grad():
            self.policy_new.load_state_dict(self.actor.state_dict()) # update policy_new to new params
        log_pi_taken_new = decentralized_pi(self.policy_new, obs, actions, actionmask, actives, self.args, self.epsilon, training=False)
        log_pi_taken_new = log_pi_taken_new * actives.squeeze(-1)

        neglogpac_new = log_pi_taken_new.sum(-1)
        pi2 = log_pi_taken.reshape(-1,self.n_agents).sum(-1)
        ratio_new = torch.exp(neglogpac_new - pi2)
        # 3.2.3 gradient for pg1 and 2
        pg_loss1 = log_pi_taken_old.view(-1,1).sum() / actives.sum()
        pg_loss2 = (adv_ex.view(-1) * ratio_new).sum() / actives.sum()
        # print(pg_loss2)

        self.policy_old.zero_grad()
        pg_loss1_grad = torch.autograd.grad(pg_loss1, self.policy_old.parameters())
        self.policy_new.zero_grad()
        pg_loss2_grad = torch.autograd.grad(pg_loss2, self.policy_new.parameters())
        # 3.2.4 calculate pg_ex_loss
        grad_total = 0
        for grad1, grad2 in zip(pg_loss1_grad, pg_loss2_grad):
            grad_total += (grad1 * grad2).sum()
            # grad_total_ = grad_total.clone()
        grad_total = grad_total.detach().item()
        target_mix_ = target_mix.reshape(obs.shape[0], -1, self.n_agents)
        pg_ex_loss = (grad_total * target_mix_).mean()

        intrinsic_loss = pg_ex_loss + self.vt_coef * v_ex_loss

        del std_adv_ex, adv_ex, pg_loss1, pg_loss2, ratio_new, neglogpac_new, pi2, pg_loss1_grad, pg_loss2_grad, grad_total, log_pi_taken_new, log_pi_taken_old, target_mix_
        torch.cuda.empty_cache()
        return intrinsic_loss
    
    
    # 학습 기록
    # @profile
    def write_scheme(self, scheme, learn_scheme, args):
        scheme["critic_losses"].append(np.mean(learn_scheme["critic_loss"])) # np.mean은 임시방편
        scheme["actor_losses"].append(learn_scheme["actor_loss"])
        scheme["intrinsic_losses"].append(learn_scheme["intrinsic_loss"])
        for i in range(0,args.num_agents):
            # scheme["intrinsic_losses"][i].append(learn_scheme["intrinsic_loss"][i])
            scheme["r_in_list"].append(learn_scheme["r_in"][i])
            # scheme["r_mix_list"][i].append(r_mix_.item())  
        # UpperAgent 학습 시퀀스도 추가할 예정 
    
    # @profile
    def write_summary(self, scheme, step, ENVargs, win_rate):
        episode, team, ep_length, interval = scheme["episode"], scheme["team"], scheme["EpisodeInfo"]["episode_length"], ENVargs.print_interval
        total_episode = np.sum(ep_length)
        total_r_ex = np.sum(scheme["EpisodeInfo"]["scores"])
        current_time = datetime.now().strftime('%m-%d %H:%M:%S')
        if self.args.train_mode==True:
            mean_r_in = [torch.mean(scheme["EpisodeInfo"]["r_in_list"][i]).item()/interval for i in range(0,self.n_agents)]
            # mean_r_mix = [np.mean(r_mix)/interval for r_mix in scheme["r_mix_list"]]
            mean_critic_loss = np.mean(scheme["EpisodeInfo"]["critic_losses"]) # if len(scheme["critic_losses"]) > 0 else 0
            mean_actor_loss = np.mean(scheme["EpisodeInfo"]["actor_losses"])
            mean_intrinsic_loss = np.mean(scheme["EpisodeInfo"]["intrinsic_losses"]) # if len(scheme["intrinsic_losses"]) > 0 else 0        
            r_in_ = [round(r,5) for r in mean_r_in]
            print(" ")
            print(f"[{current_time}] Episode {episode} team{team} ({self.algorithm}) Summary ({total_episode} step / {step} step)")
            print(f"[Reward] r_in: {r_in_}" )
            print(f"[Reward] r_ex: {total_r_ex:.4f} | win_rate: {100*win_rate:.3f}%")
            print(f"[Loss] Critic: {mean_critic_loss:.4f} | Actor: {mean_actor_loss:.4f} | Intrinsic: {mean_intrinsic_loss:.4f}")
            print(" ")
            self.writer.add_scalar("episode/reward", total_r_ex, episode)
            self.writer.add_scalar("episode/episode_length", np.mean(ep_length), episode)
            self.writer.add_scalar("episode/win_rate", win_rate*100, episode)
            self.writer.add_scalar("model/intrinsic_reward", np.mean(mean_r_in), episode)
            self.writer.add_scalar("model/critic_loss", mean_critic_loss, episode)
            self.writer.add_scalar("model/actor_loss", mean_actor_loss, episode)
            self.writer.add_scalar("model/intrinsic_loss", mean_intrinsic_loss, episode)
            scheme["EpisodeInfo"].clear()
            scheme["EpisodeInfo"]["critic_losses"], scheme["EpisodeInfo"]["actor_losses"], scheme["EpisodeInfo"]["intrinsic_losses"], scheme["EpisodeInfo"]["r_in_list"] = [], [], [], []
            # for i in range(0, self.num_agents):
            #     for epstep in range(0, scheme["EpisodeInfo"]["r_in_list"][i].shape[0]):
            #         self.writer.add_scalar(f"reward_intrinsic/{i+1}", scheme["EpisodeInfo"]["r_in_list"][i][epstep], startStep+epstep)
            del r_in_
        else:
            print(" ")
            print(f"TestStep {step - ENVargs.run_step} team{team} ({self.algorithm}) Summary ({total_episode} step / {step} step)")
            print(f"Episode {episode} | Reward: {total_r_ex:.2f} | win_rate: {100*win_rate:.3f}%")
            self.writer.add_scalar("Test/episode_length", np.mean(ep_length), episode)
            self.writer.add_scalar("Test/win_rate", win_rate*100, episode)
            self.writer.add_scalar("Test/reward", total_r_ex, episode)
            scheme["EpisodeInfo"].clear()

        scheme["EpisodeInfo"]["episode_length"], scheme["EpisodeInfo"]["scores"] = [], []

        # del mean_r_ex, mean_r_in, mean_actor_loss, mean_critic_loss, mean_intrinsic_loss, ep_length, r_in_, total_episode
        # torch.cuda.empty_cache()
    # 네트워크 모델 저장
    
    def save_model(self):
        print(" ")
        print(f"... Save Model to {self.save_path}/ckpt ...")
        obj = {}
        obj["actor"] = self.actor.state_dict()
        obj["actor_optimizer"] = self.actor_optimiser.state_dict()
        obj["critic"] = self.critic.state_dict()
        obj["critic_optimizer"] = self.critic_optimiser.state_dict()
        obj[f"intrinsic_optimizer"] = self.intrinsic_optimiser.state_dict()
        
        torch.save(obj, self.save_path+'/ckpt')
