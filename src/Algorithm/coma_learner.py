import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from modules.Critic.COMAcritic import COMAcritic
from Utils.calculate_pi import decentralized_pi
from datetime import datetime

class COMAagent:
    def __init__(self, args, actor, epsilon, save_path, load_path):  
        self.algorithm = "coma"
        self.args = args
        self.state_size = args.state_size
        self.obs_size = args.obs_size
        self.n_agents = args.num_agents
        self.n_actions = args.action_size
        self.grad_norm_clip = args.grad_norm_clip
        self.critic_training_interval = 0
        self.target_update_interval = args.target_update_interval

        self.save_path = save_path
        self.load_path = load_path
        self.device = args.device

        self.td_lambda = args.td_lambda
        self.epsilon = epsilon
        self.gamma = args.gamma

        self.critic = COMAcritic(args).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)   

        self.actor = actor
        self.actor_params = self.actor.parameters()
        self.critic_params = self.critic.parameters()

        self.memory = list()
        self.writer =SummaryWriter(save_path)

    def SetOptimiser(self):
        OptimClass = getattr(torch.optim, self.args.optimiser)
        self.actor_optimiser     = OptimClass(self.actor_params,     **self.args.optimiser_param)
        self.critic_optimiser    = OptimClass(self.critic_params,    **self.args.optimiser_param)
        if self.args.load_model == True:
            checkpoint = torch.load(self.load_path + '/ckpt', map_location=self.device)
            self.actor.load_state_dict(checkpoint[f"actor"])
            self.actor_optimiser.load_state_dict(checkpoint[f"actor_optimizer"])
            self.actor.flattenParameters()
            self.critic.load_state_dict(checkpoint["critic"])
            self.critic_optimiser.load_state_dict(checkpoint["critic_optimizer"])
            print(f"... Load Model from {self.load_path}/ckpt complete ...")

    def append_sample(self, states, obs, actions, actionmask, Reward, done, actives):
            self.memory.append((states, obs, actions, actionmask, Reward, done, actives))

    def memoryClear(self):
        self.memory.clear()
             
    def train_model(self):
        self.actor.train()
        self.critic.train()
        # extract episode data from memory
        states      = np.stack([m[0] for m in self.memory], axis=0)
        obs         = np.stack([m[1] for m in self.memory], axis=0)
        actions     = np.stack([m[2] for m in self.memory], axis=0)
        actionmask  = np.stack([m[3] for m in self.memory], axis=0)
        Reward      = np.stack([m[4] for m in self.memory], axis=0)
        done        = np.stack([m[5] for m in self.memory], axis=0)
        actives     = np.stack([m[6] for m in self.memory], axis=0)

        self.memory.clear()

        states, obs, actions, actionmask, Reward, done, actives = map(lambda x: torch.FloatTensor(x).to(self.device),[states, obs, actions, actionmask, Reward, done, actives])
        # (sequence_length, num_agent, value)
        torch.autograd.set_detect_anomaly(True)
        
        # 1. calculate critic-loss
        q_vals, target = self.train_critic(states, obs, actions, Reward, done)
        
        # 2. caluclate actor-loss
        actor_loss = self.train_actor(obs, actions, actionmask, actives, q_vals, target)   

        critic_losses = []
        for t in reversed(range(states.size(0))):
            q_t = self.critic(states, obs, actions, t).squeeze(0)
            q_taken = q_t.gather(dim=-1, index=actions[t].long())
            target_t = target[t].detach().unsqueeze(-1)
            loss = F.mse_loss(q_taken, target_t)
            self.critic_optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_params, self.grad_norm_clip)
            self.critic_optimiser.step()
            critic_losses.append(loss.item())
            del q_t, target_t, loss, q_taken
        critic_loss = np.mean(critic_losses)
        del critic_losses

        if (self.critic_training_interval+1) % self.target_update_interval == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())

        learn_scheme = {"critic_loss":critic_loss,"actor_loss":actor_loss}

        self.critic_training_interval+=1
        torch.cuda.empty_cache()
        return learn_scheme

    def train_critic(self, state, obs, action, Reward, done):
        target_q_vals = self.target_critic(state, obs, action)
        target_taken = target_q_vals.gather(dim=-1, index=action.long()).squeeze(-1)
        with torch.no_grad():
            ret = torch.zeros(target_taken.shape[0]+1, target_taken.shape[1]).to(self.device) # [bs, n_agents]
            ret[-1] = target_taken[-1] * (1 - torch.sum(done, dim=0).view(-1)) # set last value as 0
            # Backwards  recursive  update  of the "forward  view"
            for t in reversed(range(ret.shape[0]-2)):  
                ret[t] = self.td_lambda*self.gamma*ret[t+1] + Reward[t] + (1-self.td_lambda)*self.gamma*target_taken[t]*(1-done[t])
            # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
            targets = ret[:-1]
            q_vals = torch.zeros_like(target_q_vals)
            for t in reversed(range(state.shape[0])):
                q_t = self.critic(state, obs, action, t)
                q_vals[t] = q_t.squeeze(0)
                del q_t

        del ret, target_taken, target_q_vals
        torch.cuda.empty_cache()
        return q_vals.detach(), targets
    
    def train_actor(self, obs, actions, actionmask, actives, q_vals, target):
        self.actor.flattenParameters()
        hidden_state = self.actor.init_hidden()
        pi, _ = self.actor.forward(obs, hidden_state) # dhstate : (sq_length, value)
        if self.args.train_mode: # Epsilon-greedy exploration
            epsilon_action_num = pi.size(-1)
            pi = (1-self.epsilon)*pi + actionmask*(self.epsilon/epsilon_action_num)
        masked_pi = pi * actionmask

        sum_masked = masked_pi.sum(dim=-1, keepdim=True)
        masked_pi = torch.where(sum_masked == 0, torch.ones_like(sum_masked), masked_pi)
        pi = masked_pi / masked_pi.sum(dim=-1, keepdim=True)

        pi_taken = torch.gather(pi, dim=-1, index=actions.long())
        pi_active = pi_taken * actives + (1-actives) # only active agents affect to train
        log_pi_taken = torch.log(pi_active)
        
        baseline = (pi*q_vals).sum(-1).detach()
        q_taken = torch.gather(q_vals, dim=-1, index=actions.long()).squeeze(-1)

        # 2.2 calculate advantage
        advantage = (q_taken - baseline).detach()
        coma_loss = - (advantage * log_pi_taken.squeeze(-1)).mean()
        
        self.actor_optimiser.zero_grad()
        coma_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_params, self.grad_norm_clip)
        self.actor_optimiser.step()
        coma_loss_ = coma_loss.item()
        del hidden_state, pi, masked_pi, epsilon_action_num, sum_masked, pi_taken, pi_active, log_pi_taken, baseline, q_taken, advantage, coma_loss

        return coma_loss_
    
    # write train summary
    def write_scheme(self, scheme, learn_scheme, args):
        scheme["critic_losses"].append(np.mean(learn_scheme["critic_loss"]))
        scheme["actor_losses"].append(learn_scheme["actor_loss"])

    def write_summary(self, scheme, step, ENVargs, win_rate):
        episode, team, interval = scheme["episode"], scheme["team"], ENVargs.print_interval
        total_reward = np.sum(scheme["EpisodeInfo"]["scores"])/interval
        ep_length = scheme["EpisodeInfo"]["episode_length"]

        episode, team = scheme["episode"], scheme["team"]
        total_episode = np.sum(ep_length)
        if self.args.train_mode:
            mean_critic_loss = np.mean(scheme["EpisodeInfo"]["critic_losses"])
            mean_actor_loss = np.mean(scheme["EpisodeInfo"]["actor_losses"])
            current_time = datetime.now().strftime('%m-%d %H:%M:%S')
            print(" ")
            print(f"[{current_time}] Episode {episode} team{team} ({self.algorithm}) Summary ({total_episode} step / {step} step)")
            print(f"[Reward] {total_reward:.4f} | win_rate: {100*win_rate:.3f} ")
            print(f"[Loss] Critic: {mean_critic_loss:.4f} / Actor: {mean_actor_loss:.4f}" )
            print(" ")
            self.writer.add_scalar("reward/score", total_reward, episode)
            self.writer.add_scalar("episode/episode_length", np.mean(ep_length), episode)
            
            self.writer.add_scalar("episode/win_rate", 100*win_rate, episode)
            self.writer.add_scalar("model/critic_loss", mean_critic_loss, episode)
            self.writer.add_scalar("model/actor_loss", mean_actor_loss, episode)
            scheme["EpisodeInfo"].clear()
            scheme["EpisodeInfo"]["critic_losses"], scheme["EpisodeInfo"]["actor_losses"] = [], []
        else:
            print(" ")
            print(f"TestStep {step - ENVargs.run_step} team{team} ({self.algorithm}) Summary ({total_episode} step / {step} step)")
            print(f"Episode {episode} | Reward: {total_reward:.2f} | win_rate: {100*win_rate:.3f}%")
            self.writer.add_scalar("Test/episode_length", np.mean(ep_length), episode)
            self.writer.add_scalar("Test/Win_rate", win_rate*100, episode)
            self.writer.add_scalar("Test/Reward", total_reward, episode)
            scheme["EpisodeInfo"].clear()

        scheme["EpisodeInfo"]["episode_length"], scheme["EpisodeInfo"]["scores"] = [], []
        
    def save_model(self):
        print(f"... Save Model to {self.save_path}/ckpt ...")
        obj = {}
        obj[f"actor"] = self.actor.state_dict()
        obj[f"actor_optimizer"] = self.actor_optimiser.state_dict()
        obj["critic"] = self.critic.state_dict()
        obj["critic_optimizer"] = self.critic_optimiser.state_dict()
        
        torch.save(obj, self.save_path+'/ckpt')
        
    def load_model(self):
        if self.args.load_model == True:
            checkpoint = torch.load(self.load_path + '/ckpt', map_location=self.device)
            self.actor.load_state_dict(checkpoint[f"actor"])
            self.actor_optimiser.load_state_dict(checkpoint[f"actor_optimizer"])
            self.actor.flattenParameters()
            self.critic.load_state_dict(checkpoint["critic"])
            self.critic_optimiser.load_state_dict(checkpoint["critic_optimizer"])
            print(f"... Load Model from {self.load_path}/ckpt complete ...")