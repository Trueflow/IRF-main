import copy

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from modules.mixer.mixer import dmaq_Mixer # dmaq_qatten.py
from modules.intrinsic.CDS_intrinsic import Predict_Net, Combined_Predict_Net # predict_net.py
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.optim import RMSprop
from datetime import datetime


class CDSagent:
    def __init__(self, args, actor, epsilon, save_path, load_path):
        self.args = args
        self.actor = actor
        self.algorithm = "cds"
        self.writer = SummaryWriter(save_path)
        self.save_path = save_path
        self.load_path = load_path

        self.params = list(actor.parameters())

        self.training_episode = 0

        self.mixer = dmaq_Mixer(args).to(args.device)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())
        # Intrinsic Predict Net - Not use state
        self.eval_predict_withoutid = Predict_Net(args).to(args.device)
        self.target_predict_withoutid = Predict_Net(args).to(args.device)
        
        self.eval_predict_withid = Combined_Predict_Net(args).to(args.device)
        self.target_predict_withid = Combined_Predict_Net(args).to(args.device)

        self.target_predict_withid.load_state_dict(self.eval_predict_withid.state_dict())
        self.target_predict_withoutid.load_state_dict(self.eval_predict_withoutid.state_dict())

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(self.actor)

        self.n_actions = self.args.action_size
        self.memory = list()
        self.device=args.device

    def SetOptimiser(self):
        OptimClass = getattr(th.optim, self.args.optimiser)
        self.optimiser = OptimClass(self.params, **self.args.optimiser_param)
        if self.args.load_model==True: self.load_models()


    # use only team reward
    def append_sample(self, states, obs, actions, actionmask, reward, states_next, obs_next, done, actives):
        self.memory.append((states, obs, actions, actionmask, reward, states_next, obs_next, done, actives))
    
    def memoryClear(self):
        self.memory.clear()

    def train_model(self):
        intrinsic_rewards, loss, td_error_abs, hit_prob = self.train(self.actor, self.mixer, self.optimiser, self.params)
        self.memory.clear()

        learn_scheme = {
            "agent_r_in":intrinsic_rewards.reshape(self.args.num_agents, -1),
            "loss":loss,
            "td_error_abs":td_error_abs,
            "hit_prob":hit_prob
        }

        if (self.training_episode+1) % self.args.target_update_interval >= 1.0:
            self._update_targets()
        del intrinsic_rewards, loss, td_error_abs, hit_prob
        th.cuda.empty_cache()
        self.training_episode+=1
        return learn_scheme   

    def train(self, mac, mixer, optimiser, params):
        # Get the relevant quantities - my code
        states      = np.stack([m[0] for m in self.memory], axis=0)
        states_next = np.stack([m[5] for m in self.memory], axis=0)
        obs         = np.stack([m[1] for m in self.memory], axis=0)  
        obs_next    = np.stack([m[6] for m in self.memory], axis=0)  

        actions     = np.stack([m[2] for m in self.memory], axis=0)
        avail_actions = np.stack([m[3] for m in self.memory], axis=0) # actionmask
        rewards     = np.stack([m[4] for m in self.memory], axis=0) # reward
        terminated   = np.stack([m[7] for m in self.memory], axis=0) # done
        actives     = np.stack([m[8] for m in self.memory], axis=0) 
        self.memoryClear()
        # add dummy dimension to match original dimension
        states, obs, actions, avail_actions, rewards, terminated, actives, states_next, obs_next = \
            map(lambda x: th.FloatTensor(x).to(self.device).unsqueeze(0), 
            [states, obs, actions, avail_actions, rewards, terminated, actives, states_next, obs_next])
        
        actions_onehot = F.one_hot(actions.long(),num_classes=self.n_actions).squeeze(-2)
        last_actions_onehot = th.cat([th.zeros_like(actions_onehot[:, 0].unsqueeze(1)), actions_onehot[:,:-1]], dim=1)  # last_actions

        hidden_states = mac.init_hidden() # dimension : 3
        initial_hidden = hidden_states.clone().detach()

        input_here = th.cat((obs, last_actions_onehot), dim=-1).to(self.device).squeeze(0)
        mac_out, hidden_store = mac.cds_forward(input_here.clone().detach(), initial_hidden.clone().detach())
        hidden_store = hidden_store.unsqueeze(0)
        mac_out = mac_out.unsqueeze(0) #.permute(0, 2, 1, 3)
        # Pick the Q-Values for the actions taken by each agent
        
        chosen_action_qvals = th.gather(mac_out, dim=3, index=actions.long()).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out.max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions.long()).int().float()

        # Calculate the Q-Values necessary for the target
        target_hidden = self.target_mac.init_hidden()
        initial_hidden_target = target_hidden.clone().detach()
        target_mac_out, _ = self.target_mac.cds_forward(input_here.clone().detach(), initial_hidden_target.clone().detach())
        target_mac_out = target_mac_out.unsqueeze(0)
        target_mac_out = target_mac_out

        # Mask out unavailable actions (next_avail actions)
        target_mac_out[avail_actions == 0] = -9999999

        
        # Get actions that maximise live Q (for double q-learning)
        mac_out_detach = mac_out.clone().detach()
        mac_out_detach[avail_actions == 0] = -9999999
        cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
        target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        target_max_qvals = target_mac_out.max(dim=3)[0]
        target_next_actions = cur_max_actions.detach()

        cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).to(self.args.device)
        cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        
        # Intrinsic
        with th.no_grad():
            obs_intrinsic = obs.clone().permute(0, 2, 1, 3) # not use last step
            obs_intrinsic = obs_intrinsic.reshape(-1, obs_intrinsic.shape[-2], obs_intrinsic.shape[-1])
            eval_h_intrinsic = hidden_store.clone().permute(0, 2, 1, 3)
            eval_h_intrinsic = eval_h_intrinsic.reshape(-1, eval_h_intrinsic.shape[-2], eval_h_intrinsic.shape[-1])
            
            h_cat = th.cat([initial_hidden.permute(1,0,2), eval_h_intrinsic[:, :-1]], dim=1)
            add_id = th.eye(self.args.num_agents).to(obs.device)
            add_id = add_id.expand([obs.shape[0], obs.shape[1], self.args.num_agents,self.args.num_agents]).permute(0, 2, 1, 3)
            actions_onehot_clone = actions_onehot.clone().permute(0, 2, 1, 3)
            actions_onehot_clone_r = actions_onehot_clone.reshape(-1, actions_onehot_clone.shape[-2], actions_onehot_clone.shape[-1])

            intrinsic_input_1 = th.cat([h_cat, obs_intrinsic,actions_onehot_clone_r],dim=-1)
            intrinsic_input_2 = th.cat([intrinsic_input_1, add_id.reshape(-1, add_id.shape[-2], add_id.shape[-1])], dim=-1)

            intrinsic_input_1 = intrinsic_input_1.reshape(-1, intrinsic_input_1.shape[-1])
            intrinsic_input_2 = intrinsic_input_2.reshape(-1, intrinsic_input_2.shape[-1])

            next_obs_intrinsic = obs_next.permute(0, 2, 1, 3)
            next_obs_intrinsic = next_obs_intrinsic.reshape(-1, next_obs_intrinsic.shape[-2], next_obs_intrinsic.shape[-1])
            next_obs_intrinsic = next_obs_intrinsic.reshape(-1, next_obs_intrinsic.shape[-1])

            log_p_o = self.target_predict_withoutid.get_log_pi(intrinsic_input_1, next_obs_intrinsic)
            log_q_o = self.target_predict_withid.get_log_pi(intrinsic_input_2, next_obs_intrinsic, add_id.reshape([-1, add_id.shape[-1]]))

            mean_p = th.softmax(mac_out, dim=-1).mean(dim=2)
            q_pi = th.softmax(self.args.beta1 * mac_out, dim=-1)

            pi_diverge = th.cat([(q_pi[:, :, id] * th.log(q_pi[:, :, id] / mean_p)).sum(dim=-1, keepdim=True) for id in range(self.args.num_agents)],dim=-1)
            pi_diverge = pi_diverge.permute(0, 2, 1).unsqueeze(-1)

            intrinsic_rewards = self.args.beta1 * log_q_o - log_p_o
            intrinsic_rewards = intrinsic_rewards.reshape(-1, obs_intrinsic.shape[1], intrinsic_rewards.shape[-1])
            intrinsic_rewards = intrinsic_rewards.reshape(-1, obs.shape[2], obs_intrinsic.shape[1], intrinsic_rewards.shape[-1])
            intrinsic_rewards = intrinsic_rewards + self.args.beta2 * pi_diverge

        # update predict network
        add_id = add_id.reshape([-1, add_id.shape[-1]])
        for index in BatchSampler(SubsetRandomSampler(range(intrinsic_input_1.shape[0])), 256, False):
            self.eval_predict_withoutid.update(intrinsic_input_1[index], next_obs_intrinsic[index])
            self.eval_predict_withid.update(intrinsic_input_2[index], next_obs_intrinsic[index], add_id[index])

        # Mix
        ans_chosen, q_attend_regs, head_entropies = mixer(chosen_action_qvals, states, is_v=True) 
        ans_adv, _, _ = mixer(chosen_action_qvals, states, actions=actions_onehot, max_q_i=max_action_qvals, is_v=False) 
        chosen_action_qvals = ans_chosen + ans_adv

        target_chosen, _, _ = self.target_mixer(target_chosen_qvals, states_next, is_v=True)
        target_adv, _, _ = self.target_mixer(target_chosen_qvals, states_next, actions=cur_max_actions_onehot, max_q_i=target_max_qvals, is_v=False)
        target_max_qvals = target_chosen + target_adv
        
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.beta * intrinsic_rewards.mean(dim=1) + self.args.gamma*(1 - terminated)*target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        # Normal L2 loss, take mean over actual data
        loss = (td_error**2).mean() + q_attend_regs

        hit_prob = th.mean(is_max_action, dim=2).mean()

        # Optimise
        optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
        optimiser.step()

        td_error_abs = td_error.abs().mean().item()

        return intrinsic_rewards.detach(), loss.item(), td_error_abs, hit_prob.item()
    
    def write_scheme(self, scheme, learn_scheme, args):
        scheme["losses"].append(learn_scheme["loss"])
        scheme["td_error_abs"].append(learn_scheme["td_error_abs"])
        scheme["hit_prob"].append(learn_scheme["hit_prob"])
        for i in range(0,args.num_agents):
            scheme["intrinsic_rewards"].append(learn_scheme["agent_r_in"][i])
    
    def _update_targets(self):
        self.target_mac.load_state_dict(self.actor.state_dict())
        self.target_predict_withid.load_state_dict(self.eval_predict_withid.state_dict())
        self.target_predict_withoutid.load_state_dict(self.eval_predict_withoutid.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save_model(self):
        print(" ")
        print(f"... Save Model to {self.save_path}/ckpt ...")
        obj = {}
        obj["actor"] = self.actor.state_dict()
        obj["optimiser"] = self.optimiser.state_dict()
        obj["mixer"] = self.mixer.state_dict()
        obj["eval_predict_withid"] = self.eval_predict_withid.state_dict()
        obj["eval_predict_withoutid"] = self.eval_predict_withoutid.state_dict()

        obj["target_actor"] = self.target_mac.state_dict()
        obj["target_mixer"] = self.target_mixer.state_dict()
        obj["target_predict_withid"] = self.target_predict_withid.state_dict()
        obj["target_predict_withoutid"] = self.target_predict_withoutid.state_dict()

        th.save(obj, self.save_path+'/ckpt')

    def load_models(self):
        checkpoint = th.load(self.load_path + '/ckpt', map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.target_mac.load_state_dict(checkpoint["target_actor"]) # Not quite right but I don't want to save target networks
        if self.mixer is not None:
            self.mixer.load_state_dict(checkpoint["mixer"])
            self.target_mixer.load_state_dict(checkpoint["target_mixer"])
        self.optimiser.load_state_dict(checkpoint["optimiser"])
        self.eval_predict_withid.load_state_dict(checkpoint["eval_predict_withid"])
        self.target_predict_withid.load_state_dict(checkpoint["target_predict_withid"])
        self.eval_predict_withoutid.load_state_dict(checkpoint["eval_predict_withoutid"])
        self.target_predict_withoutid.load_state_dict(checkpoint["target_predict_withoutid"])
        print(f"... Load Model from {self.load_path}/ckpt complete ...")

    def write_summary(self, scheme, step, ENVargs, win_rate):
        episode, team, ep_length, interval = scheme["episode"], scheme["team"], scheme["EpisodeInfo"]["episode_length"], ENVargs.print_interval
        total_episode = np.sum(ep_length)
        total_r_ex = np.sum(scheme["EpisodeInfo"]["scores"])
        current_time = datetime.now().strftime('%m-%d %H:%M:%S')
        if self.args.train_mode==True:
            mean_r_in = [th.mean(scheme["EpisodeInfo"]["intrinsic_rewards"][i]).item()/interval for i in range(0,self.args.num_agents)]
            mean_loss = np.mean(scheme["EpisodeInfo"]["losses"]) # if len(scheme["critic_losses"]) > 0 else 0
            mean_td_error_abs = np.mean(scheme["EpisodeInfo"]["td_error_abs"])
            mean_hit_prob = np.mean(scheme["EpisodeInfo"]["hit_prob"]) # if len(scheme["intrinsic_losses"]) > 0 else 0        
            r_in_ = [round(r,5) for r in mean_r_in]
            print(" ")
            print(f"[{current_time}] Episode {episode} team{team} ({self.algorithm}) Summary ({total_episode} step / {step} step)")
            print(f"[Intrinsic Reward] r_in: {r_in_}" )
            print(f"[Reward] r_ex: {total_r_ex:.4f} | win_rate: {100*win_rate:.3f}%")
            print(f" Loss: {mean_loss:.4f} | td_error_abs: {mean_td_error_abs:.4f} | hit_prob: {mean_hit_prob:.4f}")
            print(" ")
            self.writer.add_scalar("episode/reward", total_r_ex, episode)
            self.writer.add_scalar("episode/episode_length", np.mean(ep_length), episode)
            self.writer.add_scalar("episode/win_rate", win_rate*100, episode)
            self.writer.add_scalar("model/intrinsic_reward", np.mean(mean_r_in), episode)
            self.writer.add_scalar("model/loss", mean_loss, episode)
            self.writer.add_scalar("model/td_error_abs", mean_td_error_abs, episode)
            self.writer.add_scalar("model/hit_prob", mean_hit_prob, episode)
            scheme["EpisodeInfo"].clear()
            scheme["EpisodeInfo"]["losses"], scheme["EpisodeInfo"]["hit_prob"], scheme["EpisodeInfo"]["td_error_abs"], scheme["EpisodeInfo"]["intrinsic_rewards"] = [], [], [], []
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
