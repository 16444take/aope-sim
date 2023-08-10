from typing import NamedTuple

class Config(NamedTuple):

    b_use_gpu: bool         = False             # Train on GPU
    order_type: str         = 'LM'              # Picking order data type (LM: Low Mixed, HM: High Mixed) 
    n_episodes: int         = 5000              # Number of training episodes 
    n_rollouts: int         = 64                # Number of rollouts episode 
    n_para: int             = 16                # Number of parallel environments 
    hid_unit: int           = 128               # Hidden units
    lr_actor: float         = 0.3e-3            # Actor learning rate
    lr_critic: float        = 1.0e-3            # Critic learning rate
    gamma: float            = 0.99              # Discount factor
    gamma_scale: float      = 800               # Scaling factor of gamma 
    epochs: int             = 5                 # Update epochs 
    eps_clip: float         = 0.2               # Clipping coefficient of surrogate objective loss
    vf_coef: float          = 0.5               # Coefficient of value loss
    ent_coef: float         = 0.0               # Coefficient of entropy loss
    lr_decay_pi: int        = 250               # Decay period of actor learning rate
    lr_decay_vf: int        = 250               # Decay period of critic learning rate
    gamma_pi: float         = 0.8               # Decay factor of actor learning rate
    gamma_vf: float         = 0.8               # Decay factor of critic learning rate
    b_use_gae               = False             # Use GAE for Advantage (F: Monte Calro estimator)
    b_ave_trs: bool         = True              # Average rewards with transitions
    b_na_one: bool          = False             # Elinimate one-action decision
    b_limit_kl: bool        = False             # Limit actor update via KL divergence
    target_kl: float        = 0.02              # Approximated KL div. between new and old policies 
    batch_size: int         = 64                # Mini-batch size
    nn_init_type: str       = 'orthogonal'      # How to initialize networks
    b_nrm_adv: bool         = False             # Standardize advantage
    b_use_action_mask: bool = True              # Use action mask for logP
