import copy
import torch.nn.functional as F
import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

# from modules.Critic.episodic_memory_buffer import Episodic_memory_buffer
from .vdn_Qlearner import vdn_QLearner
from modules.mixer.mixer import dmaq_Mixer # dmaq_qatten.py

class EMCagent:
    def __init__(self, args, actor, epsilon, save_path, load_path):
        self.args = args
        self.mac = actor
        self.algorithm = "emc"
        self.writer = SummaryWriter(save_path)
        self.save_path = save_path
        self.load_path = load_path

        self.params = list(self.mac.parameters())
        # Curiosity module: vdn_QLearner
        self.vdn_learner = vdn_QLearner(args, actor, save_path, load_path)
        self.mixer = dmaq_Mixer(args)
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(actor)

        self.save_buffer_cnt = 0
        self.episode_num = 0
        self.n_actions = self.args.action_size
        self.memory = list()
        self.ec_buffer = None # Episodic_memory_buffer() if args.use_emdqn else None
        self.device=args.device

    def SetOptimiser(self):
        OptimClass = getattr(th.optim, self.args.optimiser)
        self.optimiser = OptimClass(self.params, **self.args.optimiser_param)
        self.vdn_learner.SetOptimiser()
        self.cuda()
        if self.args.load_model: self.load_models()

    # use only team reward
    def append_sample(self, states, obs, actions, actionmask, reward, states_next, obs_next, done, actives):
        self.memory.append((states, obs, actions, actionmask, reward, states_next, obs_next, done, actives))
    
    def memoryClear(self):
        self.memory.clear()

    def GenerateBatch(self, memory):
        Batch = {
            "state":           th.FloatTensor(np.stack([m[0] for m in self.memory], axis=0)).to(self.device).unsqueeze(0),
            "obs":              th.FloatTensor(np.stack([m[1] for m in self.memory], axis=0)).to(self.device).unsqueeze(0), 
            "actions":          th.FloatTensor(np.stack([m[2] for m in self.memory], axis=0)).to(self.device).unsqueeze(0),
            "avail_actions":    th.FloatTensor(np.stack([m[3] for m in self.memory], axis=0)).to(self.device).unsqueeze(0),
            "reward":           th.FloatTensor(np.stack([m[4] for m in self.memory], axis=0)).to(self.device).unsqueeze(0),
            "state_next":      th.FloatTensor(np.stack([m[5] for m in self.memory], axis=0)).to(self.device).unsqueeze(0),
            "obs_next":         th.FloatTensor(np.stack([m[6] for m in self.memory], axis=0)).to(self.device).unsqueeze(0),
            "terminated":       th.FloatTensor(np.stack([m[7] for m in self.memory], axis=0)).to(self.device).unsqueeze(0),
            "actives":          th.FloatTensor(np.stack([m[8] for m in self.memory], axis=0)).to(self.device).unsqueeze(0)
        }
        Batch["actions_onehot"]=F.one_hot(Batch["actions"].long(),num_classes=self.n_actions).squeeze(-2)
        return Batch

    def train_model(self):
        batch = self.GenerateBatch(self.memory)
        intrinsic_rewards, vdn_loss = self.vdn_learner.train(batch, imac=self.mac, timac=self.target_mac)
        learn_scheme = self.sub_train(batch, self.mac, self.mixer, self.optimiser, self.params, vdn_loss, intrinsic_rewards, ec_buffer=self.ec_buffer)

        if (self.episode_num+1) % self.args.target_update_interval >= 1.0:
            self._update_targets()
        self.episode_num +=1
        self.memoryClear()
        th.cuda.empty_cache()
        return learn_scheme

    def sub_train(self, batch, mac, mixer, optimiser, params, vdn_loss, intrinsic_rewards,ec_buffer=None):
        # Get the relevant quantities, do not remove last dim
        rewards = batch["reward"]
        actions = batch["actions"]
        terminated = batch["terminated"].float()
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"]
        next_actions_onehot = th.cat([actions_onehot[:,1:],th.zeros_like(actions_onehot[:, 0].unsqueeze(1))], dim=1)

        # Calculate estimated Q-Values
        mac_hidden = mac.init_hidden()
        mac_out, _ = mac.forward(batch["obs"], mac_hidden)
        mac_out = mac_out.unsqueeze(0)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out, dim=3, index=actions.long()).squeeze(3)  

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out.max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        # Calculate the Q-Values necessary for the target: use obs (same)
        t_mac_hidden = self.target_mac.init_hidden()
        target_mac_out, _ = self.target_mac.forward(batch["obs_next"], t_mac_hidden)
        target_mac_out = target_mac_out.unsqueeze(0)
        # Mask out unavailable actions
        dummy_onehot = th.zeros_like(avail_actions[:,0].unsqueeze(1))
        dummy_onehot[..., 0] = 1
        next_avail_action = th.cat([avail_actions[:,1:], dummy_onehot], dim=1)
        target_mac_out[next_avail_action == 0] = -9999999
        # Max over target Q-Values
        # Get actions that maximise live Q (for double q-learning)
        mac_out_detach = mac_out.clone().detach()
        mac_out_detach[avail_actions == 0] = -9999999
        cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
        # add dummy Tensor to match the same dimension
        dummy_action = th.zeros_like(cur_max_actions[:,0].unsqueeze(1)).long()
        next_max_actions = th.cat([cur_max_actions[:,1:], dummy_action], dim=1) 
        target_chosen_qvals = th.gather(target_mac_out, 3, next_max_actions.long()).squeeze(3)
        target_max_qvals = target_mac_out.max(dim=3)[0]
        target_next_actions = cur_max_actions.detach()

        next_max_actions_onehot = th.zeros(next_max_actions.squeeze(3).shape + (self.n_actions,)).to(self.device)
        next_max_actions_onehot = next_max_actions_onehot.scatter_(3, next_max_actions, 1)

        # Mix
        ans_chosen, q_attend_regs, head_entropies = mixer(chosen_action_qvals, batch["state"], is_v=True)
        ans_adv, _, _ = mixer(chosen_action_qvals, batch["state"], actions=actions_onehot, max_q_i=max_action_qvals, is_v=False)
        chosen_action_qvals = ans_chosen + ans_adv

        target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state_next"], is_v=True)
        target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state_next"],actions=next_max_actions_onehot,max_q_i=target_max_qvals, is_v=False)
        target_max_qvals = target_chosen + target_adv

        targets = intrinsic_rewards+rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        # Normal L2 loss, take mean over actual data
        loss = (td_error ** 2).mean() + q_attend_regs
        if self.args.use_emdqn: # episodic dqn
            emdqn_td_error = qec_input_new.detach() - chosen_action_qvals
            emdqn_loss = (emdqn_td_error ** 2).mean() * self.args.emdqn_loss_weight
            loss += emdqn_loss

        hit_prob = th.mean(is_max_action, dim=2)
        # Optimise
        optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
        optimiser.step()

        learn_scheme = {
            "loss":loss.item(),
            "vdn_loss":vdn_loss,
            "td_error_abs":td_error.abs().mean().item(),
            "hit_prob":hit_prob.mean().item(),
            "intrinsic_rewards":intrinsic_rewards.mean().item()
        }
        return learn_scheme

    def write_scheme(self, scheme, learn_scheme, args):
        scheme["losses"].append(learn_scheme["loss"])
        scheme["vdn_losses"].append(learn_scheme["vdn_loss"])
        scheme["td_error_abs"].append(learn_scheme["td_error_abs"])
        scheme["hit_prob"].append(learn_scheme["hit_prob"])
        scheme["intrinsic_rewards"].append(learn_scheme["intrinsic_rewards"])

    def _update_targets(self):
        self.target_mac.load_state_dict(self.mac.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        print("EMC Updated target network")

    def cuda(self):
        self.mac.to(self.args.device)
        self.target_mac.to(self.args.device)
        self.vdn_learner.cuda()
        if self.mixer is not None:
            self.mixer.to(self.args.device)
            self.target_mixer.to(self.args.device)

    def save_model(self):
        print(" ")
        print(f"... Save Model to {self.save_path}/ckpt ...")
        obj={}
        obj["mac"]=self.mac.state_dict()
        obj["mixer"]=self.mixer.state_dict()
        obj["target_mac"]=self.target_mac.state_dict()
        obj["target_mixer"]=self.target_mixer.state_dict()
        obj["optimiser"]=self.optimiser.state_dict()
        obj["vmac"], obj["tvamc"], obj["svmac"], obj["prmac"], obj["vopt"], obj["propt"] = self.vdn_learner.save_models()
        th.save(obj, self.save_path+'/ckpt')

    def load_models(self):
        checkpoint = th.load(self.load_path + '/ckpt', map_location=self.device)
        self.mac.load_state_dict(checkpoint["mac"])
        # Not quite right but I don't want to save target networks
        self.target_mac.load_state_dict(checkpoint["target_mac"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer"])
        self.optimiser.load_state_dict(checkpoint["optimiser"])
        print(f"... Load Model from {self.load_path}/ckpt complete ...")

    def write_summary(self, scheme, step, ENVargs, win_rate):
        episode, team, ep_length, interval = scheme["episode"], scheme["team"], scheme["EpisodeInfo"]["episode_length"], ENVargs.print_interval
        total_episode = np.sum(ep_length)
        total_r_ex = np.sum(scheme["EpisodeInfo"]["scores"])
        current_time = datetime.now().strftime('%m-%d %H:%M:%S')
        if self.args.train_mode:
            mean_r_in = np.mean(scheme["EpisodeInfo"]["intrinsic_rewards"])
            mean_loss = np.mean(scheme["EpisodeInfo"]["losses"]) # if len(scheme["critic_losses"]) > 0 else 0
            mean_vdnloss = np.mean(scheme["EpisodeInfo"]["vdn_losses"])
            mean_td_error_abs = np.mean(scheme["EpisodeInfo"]["td_error_abs"])
            mean_hit_prob = np.mean(scheme["EpisodeInfo"]["hit_prob"]) # if len(scheme["intrinsic_losses"]) > 0 else 0        
            print(" ")
            print(f"[{current_time}] Episode {episode} team{team} ({self.algorithm}) Summary ({total_episode} step / {step} step)")
            print(f"[Reward] r_ex: {total_r_ex:.4f} | r_in: {mean_r_in:.4f} | win_rate: {100*win_rate:.3f}%")
            print(f"[Loss] ICM loss: {mean_vdnloss:.4f} | loss: {mean_loss:.4f} | td_error_abs: {mean_td_error_abs:.4f} | hit_prob: {mean_hit_prob:.4f}")
            print(" ")
            self.writer.add_scalar("episode/reward", total_r_ex, episode)
            self.writer.add_scalar("episode/episode_length", np.mean(ep_length), episode)
            self.writer.add_scalar("episode/win_rate", win_rate*100, episode)
            self.writer.add_scalar("model/intrinsic_reward", mean_r_in, episode)
            self.writer.add_scalar("model/loss", mean_loss, episode)
            self.writer.add_scalar("model/ICM_loss", mean_vdnloss, episode)
            self.writer.add_scalar("model/td_error_abs", mean_td_error_abs, episode)
            self.writer.add_scalar("model/hit_prob", mean_hit_prob, episode)
            scheme["EpisodeInfo"].clear()
            scheme["EpisodeInfo"]["losses"], scheme["EpisodeInfo"]["vdn_losses"], scheme["EpisodeInfo"]["hit_prob"], scheme["EpisodeInfo"]["td_error_abs"], scheme["EpisodeInfo"]["intrinsic_rewards"] = [], [], [], [], []

        else:
            print(" ")
            print(f"TestStep {step - ENVargs.run_step} team{team} ({self.algorithm}) Summary ({total_episode} step / {step} step)")
            print(f"Episode {episode} | Reward: {total_r_ex:.2f} | win_rate: {100*win_rate:.3f}%")
            self.writer.add_scalar("Test/episode_length", np.mean(ep_length), episode)
            self.writer.add_scalar("Test/win_rate", win_rate*100, episode)
            self.writer.add_scalar("Test/reward", total_r_ex, episode)
            scheme["EpisodeInfo"].clear()

        scheme["EpisodeInfo"]["episode_length"], scheme["EpisodeInfo"]["scores"] = [], []

