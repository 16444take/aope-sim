import random
import scipy.signal
from operator import itemgetter
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.distributions import Categorical
import torch.distributions as dist_pt
import numpy as np
import statistics
from . import config_cdsc as config
import sys
from copy import deepcopy

class RolloutBuffer():
    def __init__(self, use_gae):
        self.actions = []
        self.action_mask = []
        self.states = []
        self.vf_states = []
        self.logprobs = []
        self.rewards = []
        self.frames = []
        self.is_terminals = []
        self.returns = []    
        self.state_values = []    
        self.advantages = []    
        if use_gae: self.td_errors = []    

    def clear(self, use_gae):
        del self.actions[:]
        del self.action_mask[:]
        del self.states[:]
        del self.vf_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.frames[:]
        del self.returns[:]
        del self.state_values[:]    
        del self.advantages[:]   
        if use_gae: del self.td_errors[:]    


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_unit, use_gpu=True):
        super(Actor, self).__init__()
 
        self.STD = 2**0.5
        self.prob_min = 1e-5

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hid_unit = hid_unit

        # Set actor
        self.actor = nn.Sequential(
                            nn.Linear(state_dim, hid_unit, bias=True),
                            nn.Tanh(), 
                            nn.Linear(hid_unit, hid_unit, bias=True),
                            nn.Tanh(),
                            nn.Linear(hid_unit, action_dim, bias=True),
                            nn.Softmax(dim=-1)
                        ) 


    def initialize_weights(self, mod, nn_init_type, scale):
        for p in mod.parameters():
            if nn_init_type == "normal": p.data.normal_(0.01)
            elif nn_init_type == "xavier":
                if len(p.data.shape) >= 2: nn.init.xavier_uniform_(p.data)
                else: p.data.zero_()
            elif nn_init_type == "orthogonal":
                if len(p.data.shape) >= 2: self.orthogonal_init(p.data, gain=scale)
                else: p.data.zero_()
            else: raise ValueError("Invalid initialization error") 


    def orthogonal_init(self, tensor, gain=1):
        if tensor.ndimension() < 2: raise ValueError("Require more than two dimensions")
        rows = tensor.size(0)
        cols = tensor[0].numel()
        flattened = tensor.new(rows, cols).normal_(0, 1)
        if rows < cols: flattened.t_()
        u, s, v = torch.svd(flattened, some=True)
        if rows < cols: u.t_()
        q = u if tuple(u.shape) == (rows, cols) else v
        with torch.no_grad():
            tensor.view_as(q).copy_(q)
            tensor.mul_(gain)

        return tensor


    def forward(self):
        raise NotImplementedError
    

    def act(self, state, action_mask, b_use_action_mask, b_stochastic=True):

        act_mask_tensor = torch.Tensor(action_mask) 
        action_probs = self.actor(state)
        dist = Categorical(action_probs*act_mask_tensor)

        if b_use_action_mask:
            if b_stochastic: # Stochastic action selection 
                action = dist.sample()
                action_logprob = dist.log_prob(action)        
            else: # Deterministic action selection
                action = torch.argmax(action_probs*act_mask_tensor)
                action_logprob = dist.log_prob(action)
            return action.detach().item(),  action_logprob.detach().item()
        
        else:
            with torch.no_grad(): 
                dist_no_mask= Categorical(action_probs)
            if b_stochastic: # Stochastic action selection 
                action = dist.sample()
                action_logprob = dist_no_mask.log_prob(action) 
            else: # Deterministic action selection
                action = torch.argmax(action_probs*act_mask_tensor)
                action_logprob = dist_no_mask.log_prob(action)   
            return action.detach().item(),  action_logprob.detach().item()


    def get_old_dist(self, state, action_mask):

        with torch.no_grad():
            act_mask_tensor = torch.Tensor(action_mask) 
            action_probs = self.actor(state)
            return Categorical(action_probs*act_mask_tensor)


    def evaluate(self, state, action, b_continue_training):
        if b_continue_training:
            action_probs = self.actor(state) + 1e-5
        else:
            with torch.no_grad():
                action_probs = self.actor(state) + 1e-5 

        if torch.isnan(action_probs).any():
            for param in self.actor.parameters(): print(param)
            for param in self.critic.parameters(): print(param)
            print(state, self.state_dim, self.action_dim)
            print(state.shape, torch.isnan(state).any())

        dist = Categorical(action_probs)   
        action_logprobs = dist.log_prob(action) 
        dist_entropy = dist.entropy()

        return action_logprobs, dist_entropy


    def evaluate_mask(self, state, action, action_mask, b_continue_training):
        if b_continue_training:
            act_mask_tensor = torch.Tensor(action_mask) 
            action_probs = self.actor(state)*act_mask_tensor
        else:
            with torch.no_grad():
                act_mask_tensor = torch.Tensor(action_mask) 
                action_probs = self.actor(state)*act_mask_tensor

        if torch.isnan(action_probs).any():
            for param in self.actor.parameters(): print(param)
            for param in self.critic.parameters(): print(param)
            print(state, self.state_dim, self.action_dim)
            print(state.shape, torch.isnan(state).any())

        dist = Categorical(action_probs)   
        action_logprobs = dist.log_prob(action) 
        dist_entropy = dist.entropy()

        return action_logprobs, dist_entropy
  

class Critic(nn.Module):
    def __init__(self, value_state_dim, hid_unit, use_gpu=True):
        super(Critic, self).__init__()

        self.STD = 2**0.5
        self.hid_unit = hid_unit

        # Set critic
        self.critic = nn.Sequential(
                        nn.Linear(value_state_dim, hid_unit, bias=True),
                        nn.Tanh(), 
                        nn.Linear(hid_unit, hid_unit, bias=True),
                        nn.Tanh(),
                        nn.Linear(hid_unit, 1, bias=True)
                        )


    def initialize_weights(self, mod, nn_init_type, scale):
        for p in mod.parameters():
            if nn_init_type == "normal": p.data.normal_(0.01)
            elif nn_init_type == "xavier":
                if len(p.data.shape) >= 2: nn.init.xavier_uniform_(p.data)
                else: p.data.zero_()
            elif nn_init_type == "orthogonal":
                if len(p.data.shape) >= 2: self.orthogonal_init(p.data, gain=scale)
                else: p.data.zero_()
            else: raise ValueError("Invalid initialization error") 


    def orthogonal_init(self, tensor, gain=1):
        if tensor.ndimension() < 2: raise ValueError("Require more than two dimensions")
        rows = tensor.size(0)
        cols = tensor[0].numel()
        flattened = tensor.new(rows, cols).normal_(0, 1)
        if rows < cols: flattened.t_()
        u, s, v = torch.svd(flattened, some=True)
        if rows < cols: u.t_()
        q = u if tuple(u.shape) == (rows, cols) else v
        with torch.no_grad():
            tensor.view_as(q).copy_(q)
            tensor.mul_(gain)

        return tensor


    def forward(self):
        raise NotImplementedError


    def evaluate(self, vf_state):
        state_values = self.critic(vf_state)
        return state_values


class PPOActor:
    def __init__(self, state_dim, action_dim, config:config.Config):

        self.b_use_gpu = config.b_use_gpu
        self.n_para = config.n_para
        self.gamma = config.gamma
        self.gamma_scale = config.gamma_scale
        self.eps_clip = config.eps_clip
        self.epochs_actor = config.epochs_actor
        self.hid_unit = config.hid_unit
        self.lr_actor = config.lr_actor
        self.ent_coef = config.ent_coef
        self.batch_size = config.batch_size
        self.b_limit_kl = config.b_limit_kl
        if self.b_limit_kl: self.target_kl = config.target_kl
        self.b_ave_trs = config.b_ave_trs
        self.b_na_one = config.b_na_one
        self.b_use_gae = config.b_use_gae
        self.gae_lambda = config.gae_lambda
        self.nn_init_type = config.nn_init_type
        self.b_nrm_adv = config.b_nrm_adv
        self.lr_decay_pi = config.lr_decay_pi
        self.gamma_pi = config.gamma_pi
        self.action_dim = action_dim
        self.b_use_action_mask = config.b_use_action_mask

        # set device to cpu or cuda
        self.device = torch.device('cpu')

        if self.b_use_gpu: 
            self.device = torch.device('cuda:0') 
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(self.device)))

        self.buffer = RolloutBuffer(self.b_use_gae)
        self.policy = Actor(state_dim, action_dim, self.hid_unit).to(self.device)
        
        # NN weight and bias initialization
        for layer in self.policy.actor.modules():
            if isinstance(layer, nn.Linear): 
                layer.bias.data.fill_(0.0)
                if layer.out_features != action_dim:
                    self.policy.initialize_weights(mod=layer, nn_init_type=self.nn_init_type, scale=self.policy.STD)
                else:
                    self.policy.initialize_weights(mod=layer, nn_init_type=self.nn_init_type, scale=0.01)
                    

        self.pi_optimizer = torch.optim.Adam(params=self.policy.actor.parameters(), lr=self.lr_actor)
        self.pi_scheduler = lr_scheduler.StepLR(self.pi_optimizer, step_size=self.lr_decay_pi, gamma=self.gamma_pi)

        self.policy_old = Actor(state_dim, action_dim, self.hid_unit).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.advantage, self.loss_s, self.loss_pi = 0.0, 0.0, 0.0
        self.clip_frac, self.approx_kl, self.v_estim, self.ret_actual, self.expl_var = 0.0, 0.0, 0.0, 0.0, 0.0
        self.cum_rwd, self.ep_len = 0.0, 0.0


    def select_action(self, state, frame, action_mask, b_stochastic=True):
        with torch.no_grad():
            state_tsr = torch.as_tensor(state, dtype=torch.float32).to(self.device)
            action, action_logprob = self.policy_old.act(state_tsr, action_mask, self.b_use_action_mask, b_stochastic)
           
        self.buffer.action_mask.append(deepcopy(action_mask))        
        self.buffer.states.append(deepcopy(state))
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.frames.append(frame)

        return action


    def calc_return(self, agt_id, all_trans_agt, returns, td_errors, advantages, state_values):
        n_trans = len(returns)
        for i in range(n_trans):
            if all_trans_agt[i] == agt_id:
                self.buffer.returns.append(returns[i])
                self.buffer.td_errors.append(td_errors[i])
                if self.b_ave_trs: 
                    self.buffer.advantages.append(advantages[i]*100/n_trans)
                else: self.buffer.advantages.append(advantages[i])
                self.buffer.state_values.append(state_values[i])


    def sample(self, np_indices):
        # Minibatch sampling
        old_states = torch.as_tensor(list(itemgetter(*np_indices)(self.buffer.states)), dtype=torch.float32).to(self.device)
        old_actions = torch.as_tensor(list(itemgetter(*np_indices)(self.buffer.actions)), dtype=torch.float32).to(self.device)
        old_action_mask = torch.as_tensor(list(itemgetter(*np_indices)(self.buffer.action_mask)), dtype=torch.float32).to(self.device)
        old_logprobs = torch.as_tensor(list(itemgetter(*np_indices)(self.buffer.logprobs)), dtype=torch.float32).to(self.device)
        old_state_values = torch.as_tensor(list(itemgetter(*np_indices)(self.buffer.state_values)), dtype=torch.float32).to(self.device)

        advantages = list(itemgetter(*np_indices)(self.buffer.advantages))
        if self.b_nrm_adv:
            min_adv, max_adv = min(advantages), max(advantages)
            adv_norm = [(i - min_adv) / (max_adv - min_adv) for i in advantages]
            advantages = torch.as_tensor(adv_norm, dtype=torch.float32).to(self.device)
        else: 
            advantages = torch.as_tensor(advantages, dtype=torch.float32).to(self.device)

        return old_states, old_actions, old_action_mask, old_logprobs, advantages, old_state_values


    def actor_update(self, loss_pi): 
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()


    def update(self, agt_name):

        len_rollouts = len(self.buffer.states)        
        b_more_one_mb = True if len_rollouts > self.batch_size else False
        continue_training = True
        approx_kl_all = []

        for i in range(self.epochs_actor):

            indices = [idx for idx in range(len_rollouts)]  
            random.shuffle(indices)

            for start in range(0, len_rollouts, self.batch_size):
                minibatch_indices = np.array(indices[start:start+self.batch_size])
                if b_more_one_mb and minibatch_indices.shape[0] < self.batch_size: break

                old_states, old_actions, old_action_mask, old_logprobs, advantages, old_state_values = self.sample(minibatch_indices)                        

                # Evaluating old actions and values
                if self.b_use_action_mask:
                    logprobs, dist_entropy = self.policy.evaluate_mask(old_states, old_actions, old_action_mask, continue_training)  
                else:
                    logprobs, dist_entropy = self.policy.evaluate(old_states, old_actions, old_action_mask, continue_training)  
               
                # Calculate policy ratio (pi_theta / pi_theta__old) and surrogate loss
                ratios = torch.exp(logprobs - old_logprobs.detach())
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                loss_clip = torch.min(surr1, surr2) 

                # Calculate Clipped Objective PPO Loss                    
                loss_pi = - loss_clip.mean() - self.ent_coef*dist_entropy.mean() 
                    
                # Calculate approximate form of reverse KL Divergence for early stopping
                with torch.no_grad():
                    log_ratio = logprobs.clone() - old_logprobs 
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().item()
                if self.b_limit_kl and continue_training and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    loss_pi = loss_pi.detach()

                # Update Actor network
                if continue_training: 
                    self.actor_update(loss_pi)                        
                    approx_kl_all.append(approx_kl_div)
                continue_training = True

        clipped = ratios.gt(1+self.eps_clip) | ratios.lt(1-self.eps_clip)
        with torch.no_grad():
            self.clip_frac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            self.approx_kl = statistics.mean(approx_kl_all) 
            self.advantage = advantages.mean().item()
            self.loss_s = dist_entropy.mean().item() 
            self.loss_pi = loss_pi.mean().item()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(deepcopy(self.policy.state_dict()), strict=False)
        # clear buffer
        self.buffer.clear(self.b_use_gae)
        # step lr scheduler
        self.pi_scheduler.step()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage), strict=False)
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))




class PPOCritic:
    def __init__(self, value_state_dim, config:config.Config):

        self.b_use_gpu = config.b_use_gpu
        self.n_para = config.n_para
        self.gamma = config.gamma
        self.gamma_scale = config.gamma_scale
        self.eps_clip = config.eps_clip
        self.epochs_critic = config.epochs_critic
        self.hid_unit = config.hid_unit
        self.lr_critic = config.lr_critic
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.batch_size = config.batch_size
        self.b_limit_kl = config.b_limit_kl
        if self.b_limit_kl: self.target_kl = config.target_kl
        self.b_use_gae = config.b_use_gae
        self.gae_lambda = config.gae_lambda
        self.nn_init_type = config.nn_init_type
        self.b_nrm_adv = config.b_nrm_adv
        self.lr_decay_vf = config.lr_decay_vf
        self.gamma_vf = config.gamma_vf
        self.b_use_action_mask = config.b_use_action_mask

        # set device to cpu or cuda
        self.device = torch.device('cpu')

        if self.b_use_gpu: 
            self.device = torch.device('cuda:0') 
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(self.device)))

        self.buffer = RolloutBuffer(self.b_use_gae)
        self.critic = Critic(value_state_dim, self.hid_unit).to(self.device)
        
        # NN weight and bias initialization                    
        for layer in self.critic.critic.modules():
            if isinstance(layer, nn.Linear): 
                layer.bias.data.fill_(0.0)
                if layer.out_features != 1:
                    self.critic.initialize_weights(mod=layer, nn_init_type=self.nn_init_type, scale=self.critic.STD)
                else:
                    self.critic.initialize_weights(mod=layer, nn_init_type=self.nn_init_type, scale=1.0)

        self.vf_optimizer = torch.optim.Adam(params=self.critic.critic.parameters(), lr=self.lr_critic)
        self.vf_scheduler = lr_scheduler.StepLR(self.vf_optimizer, step_size=self.lr_decay_vf, gamma=self.gamma_vf)

        self.critic_old = Critic(value_state_dim, self.hid_unit).to(self.device)
        self.critic_old.load_state_dict(self.critic.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.loss_clip, self.loss_vf, self.loss_s, self.loss_pi = 0.0, 0.0, 0.0, 0.0
        self.clip_frac, self.approx_kl, self.v_estim, self.ret_actual, self.expl_var = 0.0, 0.0, 0.0, 0.0, 0.0
        self.cum_rwd, self.ep_len = 0.0, 0.0


    def estimate_vf(self, vf_state):
        with torch.no_grad():
            vf_state_tsr = torch.as_tensor(vf_state, dtype=torch.float32).to(self.device)
            state_value = self.critic.critic(vf_state_tsr).item()
           
        self.buffer.vf_states.append(deepcopy(vf_state))
        self.buffer.state_values.append(state_value)
        return state_value


    def sample(self, np_indices):
        # Mini-batch sampling
        old_vf_states = torch.as_tensor(list(itemgetter(*np_indices)(self.buffer.vf_states)), dtype=torch.float32).to(self.device)
        old_state_values = torch.as_tensor(list(itemgetter(*np_indices)(self.buffer.state_values)), dtype=torch.float32).to(self.device)

        returns = torch.as_tensor(list(itemgetter(*np_indices)(self.buffer.returns)), dtype=torch.float32).to(self.device)

        return old_vf_states, old_state_values, returns


    def calc_vloss(self, state_values, returns):
        return self.MseLoss(returns, state_values) 


    def critic_update(self, loss_vf): 
        loss_vf_scaled = self.vf_coef * loss_vf
        self.vf_optimizer.zero_grad()
        loss_vf_scaled.mean().backward()
        self.vf_optimizer.step()


    def update(self):

        len_rollouts = len(self.buffer.returns)
        b_more_one_mb = True if len_rollouts > self.batch_size else False
        continue_training = True
        for i in range(self.epochs_critic):

            indices = [idx for idx in range(len_rollouts)]  
            random.shuffle(indices)

            for start in range(0, len_rollouts, self.batch_size):
                minibatch_indices = np.array(indices[start:start+self.batch_size])
                if b_more_one_mb and minibatch_indices.shape[0] < self.batch_size: break
                old_vf_states, old_state_values, returns = self.sample(minibatch_indices)                        
                # Evaluating values
                state_values = self.critic.evaluate(old_vf_states)  
                # Match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                # Calculate Value Loss
                loss_v = self.calc_vloss(state_values, returns)
                # Update Critic networks
                self.critic_update(loss_v)                        

        with torch.no_grad():
            self.v_estim = state_values.mean().item() 
            self.loss_vf = loss_v.mean().item() 
            self.ret_actual = returns.mean().item()
            if self.b_use_gpu: 
                self.expl_var = self.explained_variance(state_values.cpu().detach().numpy().flatten(), returns.cpu().detach().numpy().flatten())
            else:
                self.expl_var = self.explained_variance(state_values.detach().numpy().flatten(), returns.detach().numpy().flatten())

        # Copy new weights into old critic
        self.critic_old.load_state_dict(deepcopy(self.critic.state_dict()), strict=False)
        # clear buffer
        self.buffer.clear(self.b_use_gae)
        # step lr scheduler
        self.vf_scheduler.step()
    
    
    def save(self, checkpoint_path):
        torch.save(self.critic_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.critic.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage), strict=False)
        self.critic_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


    def explained_variance(self, y_pred: np.ndarray, y_true: np.ndarray):
        var_y = np.var(y_true)
        return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y    
        
       


