import copy
import torch as th
import torch.nn.functional as func
from modules.Actor import REGISTRY as mac_REGISTRY
import numpy as np

class vdn_QLearner: # intrinsic module of EMC
    def __init__(self, args, actor, save_path, load_path):
        self.args = args
        self.mac = copy.deepcopy(actor)
        self.target_mac = copy.deepcopy(actor)
        self.soft_target_mac = copy.deepcopy(actor)

        self.params = list(self.mac.parameters())

        self.episode_num = 0
        self.predict_mac = mac_REGISTRY["rnn"](args, args.obs_size, args.action_size)
        self.predict_params = list(self.predict_mac.parameters())

        self.decay_stats_t = 0
        self.state_shape = args.state_size
        self.n_actions = args.action_size
        self.n_agents = args.num_agents
        #self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        #self.predict_optimiser = Adam(params=self.predict_params, lr=args.lr)
        self.load_path = load_path
        self.device = args.device
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC

    def SetOptimiser(self):
        OptimClass = getattr(th.optim, self.args.optimiser)
        self.optimiser = OptimClass(self.params, **self.args.optimiser_param)
        self.predict_optimiser = OptimClass(self.params, **self.args.optimiser_param)
        if self.args.load_model: self.load_models()

    def train(self, batch, imac=None, timac=None):
        #self.SetOptimiser()
        intrinsic_rewards, vdn_loss = self.subtrain(batch, self.mac, imac=imac, timac=timac)
        self._smooth_update_predict_targets()
        if (self.episode_num+1) % self.args.target_update_interval >= 1.0:
            self._update_targets()
        self.episode_num+=1
        return intrinsic_rewards, vdn_loss

    # EMC - Intrinsic Module
    def subtrain(self, batch, mac, imac=None, timac=None):
        # Get the relevant quantities. batch: self.memory
        rewards = batch["reward"]
        actions = batch["actions"]
        terminated = batch["terminated"].float()#
 
        avail_actions = batch["avail_actions"]
        batch_size = rewards.size(1)

        # Calculate estimated Q-Values
        mac_hidden = mac.init_hidden()

        target_mac_out, _ = self.target_mac.forward(batch["obs_next"], mac_hidden)
        soft_target_mac_out, _ = self.soft_target_mac.forward(batch["obs_next"], mac_hidden)

        mac_out, _ = mac.forward(batch["obs"], mac_hidden)
        predict_mac_out, _ = self.predict_mac.forward(batch["obs_next"], mac_hidden)
        mac_out, target_mac_out, soft_target_mac_out, predict_mac_out = mac_out.unsqueeze(0), target_mac_out.unsqueeze(0),  soft_target_mac_out.unsqueeze(0), predict_mac_out.unsqueeze(0)

        soft_target_mac_out_next = soft_target_mac_out.clone().detach()
        soft_target_mac_out_next = soft_target_mac_out_next.contiguous().view(-1, self.n_actions) * 10

        predict_mac_out = predict_mac_out.contiguous().view(-1, self.n_actions)

        prediction_error = func.pairwise_distance(predict_mac_out, soft_target_mac_out_next, p=2.0, keepdim=True)
        prediction_error = prediction_error.reshape(batch_size, -1, self.n_agents)

        intrinsic_rewards = self.args.curiosity_scale * (prediction_error.mean(dim=-1, keepdim=True).detach())

        prediction_loss = prediction_error.mean()

        self.predict_optimiser.zero_grad()
        prediction_loss.backward()
        predict_grad_norm = th.nn.utils.clip_grad_norm_(self.predict_params, self.args.grad_norm_clip)
        self.predict_optimiser.step()

        ############################

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out, dim=3, index=actions.long()).squeeze(3)

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out.max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        # Max over target Q-Values
        # Get actions that maximise live Q (for double q-learning)
        mac_out_detach = mac_out.clone().detach()
        mac_out_detach[avail_actions == 0] = -9999999
        cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
        # add dummy Tensor to match the same dimension
        dummy_action = th.zeros_like(cur_max_actions[:,0].unsqueeze(1)).long()
        next_max_actions = th.cat([cur_max_actions, dummy_action], dim=1)
        target_max_qvals = th.gather(target_mac_out, 3, next_max_actions).squeeze(3)
        # mixer - only have sum
        chosen_action_qvals = th.sum(chosen_action_qvals, dim=2, keepdim=True)
        target_max_qvals = th.sum(target_max_qvals, dim=2, keepdim=True)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        # Normal L2 loss, take mean over actual data
        loss = td_error.mean()

        hit_prob = th.mean(is_max_action, dim=2)

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        
        if self.args.curiosity_decay_rate <= 1.0:
            if self.args.curiosity_scale > self.args.curiosity_decay_stop:
                    self.args.curiosity_scale = self.args.curiosity_scale * self.args.curiosity_decay_rate
            else:
                    self.args.curiosity_scale = self.args.curiosity_decay_stop
        else:
                if self.args.curiosity_scale < self.args.curiosity_decay_stop:
                    self.args.curiosity_scale = self.args.curiosity_scale * self.args.curiosity_decay_rate
                else:
                    self.args.curiosity_scale = self.args.curiosity_decay_stop


        return intrinsic_rewards, loss.item()

    def _update_targets(self):
        self.target_mac.load_state_dict(self.mac.state_dict())
        print("Vdn Updated target network")

    def _smooth_update_predict_targets(self):
        self.soft_update(self.soft_target_mac, self.mac, self.args.soft_update_tau)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.to(self.args.device)
        self.target_mac.to(self.args.device)
        self.soft_target_mac.to(self.args.device)
        self.predict_mac.to(self.args.device)

    def save_models(self):
        vmac = self.mac.state_dict()
        tvmac = self.target_mac.state_dict()
        svmac = self.soft_target_mac.state_dict()
        prmac = self.predict_mac.state_dict()
        vopt = self.optimiser.state_dict()
        propt = self.predict_optimiser.state_dict()
        return vmac, tvmac, svmac, prmac, vopt, propt

    def load_models(self):
        checkpoint = th.load(self.load_path + '/ckpt', map_location=self.device)
        self.mac.load_state_dict(checkpoint["vmac"])
        # Not quite right but I don't want to save target networks
        self.target_mac.load_state_dict(checkpoint["tvamc"])
        self.soft_target_mac.load_state_dict(checkpoint["svmac"])
        self.predict_mac.load_state_dict(checkpoint["prmac"])
        
        self.optimiser.load_state_dict(checkpoint["vopt"])
        self.predict_optimiser.load_state_dict(checkpoint["propt"])
        
       
