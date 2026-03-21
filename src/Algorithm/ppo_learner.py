import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from modules.Critic.PPOcritic import PPOcritic
from Utils.calculate_pi import decentralized_pi_ppo
from datetime import datetime

class PPOagent:
    def __init__(self, args, actor, save_path, load_path):
        self.args = args
        self.save_path = save_path
        self.load_path = load_path
        self.device = args.device

        self.n_agents = args.num_agents
        self._lambda = args._lambda
        self.gamma = args.gamma
        self.mu = args.mu
        self.eps_clip = args.eps_clip

        self.memory = []
        self.writer =SummaryWriter(save_path)
        self.actor = actor
        self.policy_old = copy.deepcopy(self.actor)
        self.critic = PPOcritic(args).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.training_interval = 0
        self.target_update_interval = args.target_update_interval

        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=args.OPTlr)
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=args.OPTlr)
        self.grad_norm_clip = args.grad_norm_clip

        if args.load_model == True:
            print(f"... Load Model from {self.load_path}/ckpt ...")
            checkpoint = torch.load(self.load_path + '/ckpt', map_location=self.device)
            for i in range(self.n_agents):
                self.actors[i].load_state_dict(checkpoint[f"actor_{i}"])
                self.actor_optimizers[i].load_state_dict(checkpoint[f"actor_optimizer_{i}"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def append_sample(self, states, actions, Groupreward, reward, done, actionMask):
        self.memory.append((states, actions, Groupreward, reward, done, actionMask))

    def train_model(self):
        self.actor.train()
        self.critic.train()

        last_item = self.memory[-1] # 다음 학습 시 역전파를 위해 맨 마지막 값만 저장

        # 메모리에서 필요한 인자 받아오기
        states      = np.stack([m[0] for m in self.memory], axis=0)
        actions     = np.stack([m[1] for m in self.memory], axis=0)
        Groupreward = np.stack([m[2] for m in self.memory], axis=0)
        reward      = np.stack([m[3] for m in self.memory], axis=0)
        done        = np.stack([m[4] for m in self.memory], axis=0)
        actionMask  = np.stack([m[5] for m in self.memory], axis=0)

        self.memory.clear()
        self.memory.append(last_item) # 이 last_item은 target 계산에만 쓰임
        # states = torch.FloatTensor(states).to(self.device).unsqueeze(0)
        # actions = torch.FloatTensor(actions).to(self.device).unsqueeze(0)
        # Groupreward = torch.FloatTensor(Groupreward).to(self.device).unsqueeze(0)
        # reward = torch.FloatTensor(reward).to(self.device).unsqueeze(0)
        # done = torch.FloatTensor(done).to(self.device).unsqueeze(0)
        # actionMask = torch.FloatTensor(actionMask).to(self.device).unsqueeze(0)
        states, actions, Groupreward, rewards, done, actionMask = map(lambda x: torch.FloatTensor(x).to(self.device).unsqueeze(0),[states, actions, Groupreward, reward, done, actionMask])

        torch.autograd.set_detect_anomaly(True)

        old_state_values = self.target_critic(states)
        state_values = self.critic(states)
        # print(f"state : {states.shape} | action : {actions.shape} | groupReward : {Groupreward.shape} | reward : {rewards.shape} | done : {done.shape} | actionMask : {actionMask.shape}")
        rewards = Groupreward + self.mu * rewards
        # rewards = (rewards - rewards.mean()) / (rewards.std()s + 1e-7) # normalize
        advantages = rewards.detach() - old_state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # gae 방식으로 value target 계산
        with torch.no_grad():
            target = state_values.new_zeros(*state_values.shape).to(self.device)
            target[:,-1] = state_values[:,-1]
            for t in reversed(range(target.shape[1]-2)):
                target[:,t] = rewards[:,t] + self.gamma*state_values[:,t+1]*(1-done[:,t]) - state_values[:,t] + \
                    self.gamma*self._lambda*(1-done[:,t])*target[:,t+1]
        # calculate critic loss
        critic_loss = F.mse_loss(state_values[:,:-1], target[:,:-1])

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip)
        self.critic_optimizer.step()
        
        old_logprobs, _ = decentralized_pi_ppo(self.policy_old, states[:,:-1,0], actions[:,:-1,0], actionMask[:,:-1,:], self.args, training=False)
        logprobs, dist_entropy = decentralized_pi_ppo(self.actor, states[:,:-1,0], actions[:,:-1,0], actionMask[:,:-1,:], self.args)
        if not torch.isfinite(logprobs).any():
            print("Non-finite detected in logprobs")  
        if not torch.isfinite(old_logprobs).any():
            print("Non-finite detected in old_logprobs")   
        # Finding the ratio (pi_theta / pi_theta__old)
        ratios = torch.exp(logprobs - old_logprobs.detach())
        # print(f"Ratios min: {ratios.min().item()}, max: {ratios.max().item()}")
        # Finding Surrogate Loss  
        # 생각해보니까 adv도 행렬로 나오는데?
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0-self.eps_clip, 1.0+self.eps_clip) * advantages

        # final loss of clipped objective PPO
        actor_loss = -torch.mean(torch.min(surr1, surr2)) - 0.01 * torch.mean(dist_entropy)
        # + 0.5 * F.mse_loss(state_values[:,:-1], rewards[:,:-1]) 도 들어가지만, 앞서 critic_loss를 구했으니 생략
            
        # take gradient step
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm_clip)
        self.actor_optimizer.step()

        if (self.training_interval+1) % self.target_update_interval == 0:
            self.policy_old.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
        
        learn_scheme = {
            "critic_loss":critic_loss.detach().item(),
            "actor_loss":actor_loss.detach().item()
        }
        torch.cuda.empty_cache()
        return learn_scheme
    
    def write_scheme(self, scheme, learn_scheme):
        scheme["critic_losses"].append(learn_scheme["critic_loss"])
        scheme["actor_losses"].append(learn_scheme["actor_loss"])

    def write_summary(self,scheme, upperscheme, args, win_rate):
        episode, team, ep_length, interval = scheme["episode"], scheme["team"], scheme["episode_length"], args.print_interval

        mean_critic_loss = np.mean(upperscheme["critic_losses"])
        mean_actor_loss = np.mean(upperscheme["actor_losses"])
        if args.train_mode:
            print("-------------------------------------------------------------------------------------------------------------")
            print(f"team{team} UpperAgent [Loss] Critic: {mean_critic_loss:.5f} | Actor : {mean_actor_loss:.5f}")
            self.writer.add_scalar("reward/score", np.sum(scheme["scores"])/interval, episode)
            self.writer.add_scalar("episode/episode_length", np.mean(ep_length), episode)
            self.writer.add_scalar("episode/win_rate", win_rate, episode)
            self.writer.add_scalar("model/critic_loss", mean_critic_loss, episode)
            self.writer.add_scalar("model/actor_loss", mean_actor_loss, episode)
        upperscheme["critic_losses"], upperscheme["actor_losses"] = [], []

    def save_model(self):
        print(f"... Save Model to {self.save_path}/ckpt ...")
        obj = {}
        obj["critic"] = self.critic.state_dict()
        obj["actor"] = self.actor.state_dict()
        obj["critic_optimizer"] = self.critic_optimizer.state_dict()
        obj["actor_optimizer"] = self.actor_optimizer.state_dict()
