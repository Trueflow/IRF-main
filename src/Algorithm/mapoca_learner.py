import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from modules.Critic.POCAcritic import POCAcritic
from datetime import datetime

class POCAagent:
    def __init__(self, args, actor, epsilon, save_path, load_path):
        self.args = args
        self.algorithm = "MA-POCA"
        self.num_agents = args.num_agents
        self.discount_factor = args.gamma
        self.td_lambda = args.td_lambda
        self.epsilon = epsilon
        self.clip_eps = args.clip_eps
        self.mu = args.mu
        self.grad_norm_clip = args.grad_norm_clip
        self.device = args.device
        self.save_path = save_path
        self.load_path = load_path

        self.actor = actor
        self.critic = POCAcritic(args).to(self.device)
        self.policy_old = copy.deepcopy(actor)

        self.target_training_interval = 0
        self.target_update_interval = args.target_update_interval
        self.memory = list()
        self.writer = SummaryWriter(save_path)
    
    def SetOptimiser(self):
        OptimClass = getattr(torch.optim, self.args.optimiser)
        self.actor_optimiser = OptimClass(self.actor.parameters(), **self.args.optimiser_param)
        self.critic_optimiser = OptimClass(self.critic.parameters(), **self.args.optimiser_param)
        
        if self.args.load_model:
            checkpoint = torch.load(self.load_path + '/ckpt', map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.actor_optimiser.load_state_dict(checkpoint["actor_optimizer"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.critic_optimiser.load_state_dict(checkpoint["critic_optimizer"])
            print(f"... Load Model from {self.load_path}/ckpt complete ...")
    
    def save_model(self):
        print(f"... Save Model to {self.save_path}/ckpt ...")
        obj = {}
        obj["actor"] = self.actor.state_dict()
        obj["actor_optimiser"] = self.actor_optimiser.state_dict()
        obj["critic"] = self.critic.state_dict()
        obj["critic_optimiser"] = self.critic_optimiser.state_dict()
        torch.save(obj, self.save_path+'/ckpt')

    def append_sample(self, states, obs, actions, actionmask, reward, next_states, next_obs, done, actives):
        self.memory.append((states, obs, actions, actionmask, reward, next_states, next_obs, done, actives))

    def train_model(self):
        self.actor.train()
        self.critic.train()
        # extract episode data from memory
        states      = np.stack([m[0] for m in self.memory], axis=0)
        obs         = np.stack([m[1] for m in self.memory], axis=0)
        actions     = np.stack([m[2] for m in self.memory], axis=0)
        actionmask  = np.stack([m[3] for m in self.memory], axis=0)
        reward      = np.stack([m[4] for m in self.memory], axis=0)  
        next_states = np.stack([m[5] for m in self.memory], axis=0)  
        next_obs    = np.stack([m[6] for m in self.memory], axis=0)  
        done        = np.stack([m[7] for m in self.memory], axis=0)  
        actives     = np.stack([m[8] for m in self.memory], axis=0)  
        self.memory.clear()
        torch.cuda.empty_cache()

        states, obs, actions, actionmask, reward, next_states, done, actives = map(lambda x: torch.FloatTensor(x).to(self.device),
                                                                              [states, obs, actions, actionmask, reward, next_states, done, actives])

        ret, prob_olds = self.compute_advantages_and_returns(states, obs, actions, actives, reward, next_states, done, actionmask)
        seq_length = states.shape[0]
        # Train start
        prob = self.calculate_pi(self.actor, obs, actions, actionmask, training=True) # log_pi
        ratio = torch.exp(prob - prob_olds)
        q = self.compute_critic_q(states, obs, actions)

        ret_broadcast = ret.unsqueeze(1).expand(seq_length, self.num_agents, 1)  # (seq_len, num_agents, 1)

        adv = ret_broadcast - q
        adv_std = adv.std()
        if adv_std<1e-12 or torch.isnan(adv_std):
            n_adv = adv - adv.mean()
        else:
            n_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        surr1 = ratio * n_adv.detach() 
        surr2 = torch.clamp(ratio, min=1-self.clip_eps, max=1+self.clip_eps) * n_adv.detach()
        # 클리핑된 두 번째 손실 항목 계산
        actor_loss = (- torch.min(surr1, surr2) * actives).sum() / (actives.sum() + 1e-8)  

        # update actor
        self.actor_optimiser.zero_grad()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm_clip)
        actor_loss.backward()  
        self.actor_optimiser.step()
        actor_loss_value = actor_loss.item()

        baseline_loss = torch.mean(adv**2)
        
        del prob, ratio, q, n_adv, surr1, surr2, actor_loss
        torch.cuda.empty_cache()

        value = self.critic(states, obs)  # shape: (sequence, 1)
        critic_loss = F.mse_loss(value, ret).mean() + self.mu * baseline_loss

        # update critic
        self.critic_optimiser.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip)
        critic_loss.backward()
        self.critic_optimiser.step()
        critic_loss_value = critic_loss.item()

        del value, ret, adv, critic_loss, baseline_loss
        del states, actions, actionmask, reward, next_states, done, actives
        torch.cuda.empty_cache()
        
        learn_scheme = {
            "actor_loss" : actor_loss_value,  
            "critic_loss" : critic_loss_value
        }
        
        if (self.target_training_interval+1) % self.target_update_interval == 0:
            self.update_policy_old()
        self.target_training_interval += 1
        return learn_scheme

    def compute_advantages_and_returns(self, states, obs, actions, actives, reward, next_states, done, actionmask):
        max_t = states.shape[0]  # sequence_length
        with torch.no_grad():
            value = self.critic(states, obs)
            next_value = self.critic(next_states, obs)
            delta = reward + ((1 - done) * self.discount_factor * next_value) - value

            adv = delta.clone()
            # initialize TD error as advantage
            adv, done = map(lambda x: x.view(max_t, -1).transpose(0,1).contiguous(), [adv, done])

            for t in reversed(range(max_t-1)):
                adv[:,t] += (1 - done[:,t]) * self.discount_factor * self.td_lambda * adv[:,t+1]

            adv = adv.transpose(0,1).contiguous().view(-1, 1)

            ret = adv + value

            prob_olds = torch.zeros_like(actions)

            pi_old_taken = self.calculate_pi(self.policy_old, obs, actions, actionmask, False) #이전 정책 확률 저장
            prob_olds = pi_old_taken
            del pi_old_taken
            torch.cuda.empty_cache()
        return ret, prob_olds
    
    def compute_critic_q(self, states, obs, actions):
        q_list = []
        for i in range(0,self.num_agents):
            q = self.critic.compute_q(states, obs, actions, i)
            q_list.append(q)
        q_ = torch.stack(q_list, dim=1) # (seq_len, num_agents, 1)
        del q_list, q
        torch.cuda.empty_cache()
        return q_

    # calculate log pi
    def calculate_pi(self, policy, obs, actions, actionmask, training=False):
        policy.flattenParameters() 
        hidden_state = policy.init_hidden()

        pi, _ = policy.forward(obs, hidden_state)
        
        if len(pi.shape) == 3:  # (sequence_length, batch_size, action_size)
            pi = pi.squeeze(1)  # (sequence_length, action_size)

        masked_pi = pi * actionmask
        
        sum_masked = masked_pi.sum(dim=-1, keepdim=True)
        mask_normalized = actionmask / (actionmask.sum(dim=-1, keepdim=True) + 1e-8)
        masked_pi = torch.where(sum_masked == 0, mask_normalized, masked_pi)
        
        pi = masked_pi / (masked_pi.sum(dim=-1, keepdim=True) + 1e-8)
        pi_taken = torch.gather(pi, dim=-1, index=actions.long())
        log_pi_taken = torch.log(pi_taken + 1e-8)

        del pi, masked_pi, sum_masked, mask_normalized, pi_taken
        torch.cuda.empty_cache()
        return log_pi_taken

    def update_policy_old(self):
        self.policy_old.load_state_dict(self.actor.state_dict())

    def write_scheme(self, scheme, learn_scheme, args):
        scheme["critic_losses"].append(learn_scheme["critic_loss"])
        scheme["actor_losses"].append(learn_scheme["actor_loss"])

    # write train summary
    def write_summary(self, scheme, step, ENVargs, win_rate=0):
        episode, team, ep_length, interval = scheme["episode"], scheme["team"], scheme["EpisodeInfo"]["episode_length"], ENVargs.print_interval
        total_episode, total_score = np.sum(ep_length), np.sum(scheme["EpisodeInfo"]["scores"])
        current_time = datetime.now().strftime('%m-%d %H:%M:%S')
        if self.args.train_mode:
            critic_loss, actor_loss = np.mean(scheme["EpisodeInfo"]["critic_losses"]), np.mean(scheme["EpisodeInfo"]["actor_losses"])
            print(f"\n[{current_time}] Episode {episode} team{team} ({self.algorithm}) Summary ({total_episode} step / {step} step)")
            print(f"[Reward] Total Reward: {total_score:.4f} | win_rate: {100*win_rate:.3f}%" )
            print(f"[Loss] Critic: {critic_loss:.4f} / Actor: {actor_loss:.4f}")
            self.writer.add_scalar("episode/episode_length", np.mean(ep_length), episode)
            self.writer.add_scalar("episode/reward", total_score, episode)
            self.writer.add_scalar("episode/win_rate", 100*win_rate, episode)
            self.writer.add_scalar("model/actor_loss", actor_loss, episode)
            self.writer.add_scalar("model/critic_loss", critic_loss, episode)
            scheme["EpisodeInfo"].clear()
            scheme["EpisodeInfo"]["critic_losses"], scheme["EpisodeInfo"]["actor_losses"] = [], []
        else:
            print(f"\nTestStep {step - ENVargs.run_step} team{team} ({self.algorithm}) Summary ({total_episode} step / {step} step)")
            print(f"Episode {episode} | Reward: {total_score:.2f} | win_rate: {100*win_rate:.3f}%")
            self.writer.add_scalar("Test/episode_length", np.mean(ep_length), episode)
            self.writer.add_scalar("Test/Win_rate", 100*win_rate, episode)
            self.writer.add_scalar("Test/Reward", total_score, episode)
            scheme["EpisodeInfo"].clear()

        scheme["EpisodeInfo"]["scores"], scheme["EpisodeInfo"]["episode_length"]= [],[]
        
    def memoryClear(self):
        self.memory.clear()
        torch.cuda.empty_cache()