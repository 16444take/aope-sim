import random
import time
import sys
import datetime
import argparse
import os
import csv
import numpy as np
import math
from copy import deepcopy
import pandas as pd
from operator import itemgetter
import shutil
import ray
import collections

from sim_env.aope_app import AOPE_App

# Settings for RL 
import torch
from ppo.illr import PPO
import ppo.config_illr as ppo_config

if __name__ == '__main__':

    cur_dir = os.path.join(os.environ['USERPROFILE'])
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

    save_dir = 'log_train'
    now = datetime.datetime.now()
    timestamp = now.isoformat(timespec='seconds').replace('-','').replace(':','').replace('.','').replace('T','_')[2:]
    save_dir = '{}\\{}\\'.format(save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # set random seed
    n_seed = 1
    torch_seed = 1
    torch.manual_seed(torch_seed)

    rl_config = ppo_config.Config()

    # Order data type
    order_type = rl_config.order_type 

    # Read picking order data 
    if order_type == 'LM': fpath = '.\\order_data\\LowMixedOrder.csv'
    elif order_type == 'HM':  fpath = '.\\order_data\\HighMixedOrder.csv'

    df = pd.read_csv(fpath, sep=',') 
    uniq_ship = df['ship_id'].unique().tolist()       # List of unique shipping destinations 
    uniq_type = df['product_id'].unique().tolist()    # List of unique item types    
    n_ship = len(uniq_ship)                           # Number of total shipping destinations
    n_type = len(uniq_type)                           # Number of total item types

    # Save python files
    self_name = os.path.basename(__file__)
    file_names = [self_name, 'ppo\\illr.py','ppo\\config_illr.py']
    for i in range(len(file_names)): 
        rev_name = file_names[i].replace('ppo\\','').replace('.py','_copied.py')
        shutil.copy('.\\'+file_names[i], save_dir+'\\'+rev_name)
        
    n_save_model_ep = 25 # Set data save periods

    t_max = 2e5
    n_episode = rl_config.n_episodes
    n_repeat_ep = int(rl_config.n_rollouts / rl_config.n_para)  

    agt_names = ['FR','PC','PR1','PR2']
    obs_dims = [7,28,27,119]

    # Read item-location data
    fpath_item_ones_place = '.\\order_data\\ones_place_'+order_type+'_rs2.csv'
    fpath_rand_uniq_ship = '.\\order_data\\rand_uniq_ship_'+order_type+'_rs2.csv'

    f_item_ones_place = open(fpath_item_ones_place, 'r')
    l_temp_item_ones_place = f_item_ones_place.readlines()
    l_item_ones_place = []
    for xx in l_temp_item_ones_place: l_item_ones_place.append(int(xx.rstrip('\n')))
    f_rand_uniq_ship = open(fpath_rand_uniq_ship, 'r')
    l_temp_rand_uniq_ship = f_rand_uniq_ship.readlines()
    l_rand_uniq_ship = []
    for xx in l_temp_rand_uniq_ship: l_rand_uniq_ship.append(xx.rstrip('\n'))

    agt_name, obs_dim, act_dim = ['']*4, [0]*4, [0]*4
    result_file, model_fname, log_result = ['']*4, ['']*4, ['']*4
    
    log_tr_header = []
    rl_agents = [] 
    for learner in range(len(agt_names)):
        if learner == 0: agt_name[learner], obs_dim[learner], act_dim[learner] = 'FR', obs_dims[0], n_ship
        elif learner == 1: agt_name[learner], obs_dim[learner], act_dim[learner] = 'PC', obs_dims[1], n_type
        elif learner == 2: agt_name[learner], obs_dim[learner], act_dim[learner] = 'PR1', obs_dims[2], 3
        elif learner == 3: agt_name[learner], obs_dim[learner], act_dim[learner] = 'PR2', obs_dims[3], 4

        model_fname[learner] = os.path.join(save_dir, 'model_'+agt_name[learner]+'_rs'+str(n_seed)+'_ts'+str(torch_seed)+'_ep')

        # All-episode log
        result_file[learner] =  os.path.join(save_dir, 'rl_metrics')
        log_result[learner] = open(result_file[learner] +'_'+agt_name[learner]+'.csv', 'w')
        log_header = 'step,return,ret_std,ep_len,ep_std,loss_pi,advantage,loss_vf,loss_s,clip_frac,approx_kl,ret_actual,v_estim,expl_var'
        
        for i in range(rl_config.n_rollouts): log_header += ',ret_rol'+str(i+1)
        if learner == 0:
            for i in range(rl_config.n_rollouts): log_header += ',cum_rol'+str(i+1)
        print(log_header, file=log_result[learner])

        # Set RL agent
        rl = PPO(state_dim = obs_dims[learner] + act_dim[learner] + 1, 
                 action_dim = act_dim[learner],
                 config = rl_config)
        rl.save(model_fname[learner]+'0.pt') # Save initial model
        rl_agents.append(rl)                                                        

    ray.init()       
    train_envs =[AOPE_App.remote(n_seed = random.randint(1, 100000), 
                            marl_type = 'illr',
                            rl = rl_agents,
                            rlCritic = None,
                            order_type = order_type,
                            l_rand_uniq_ship = l_rand_uniq_ship,
                            l_item_ones_place = l_item_ones_place) 
                  for _ in range(rl.n_para)]

    ret_max = -1000
    start_time = time.perf_counter()
    for i in range(n_episode+1):            
        l_cum_rwd, l_ep_len = [], []
        l_ep_ret = [[], [], [], []]
        for q in range(n_repeat_ep):
            # Run one episode
            ray.get([train_envs[j].run_episode.remote(t_max) for j in range(rl_agents[0].n_para)])
            time.sleep(0.01)
            for k in range(len(agt_names)):
                for j in range(len(train_envs)):
                    temp_buffer, cum_rwd, ep_len, ep_ret = ray.get(train_envs[j].get_history.remote(k)) 
                    if k == 0: l_cum_rwd.append(cum_rwd), l_ep_len.append(ep_len)
                    l_ep_ret[k].append(ep_ret)
                    rl_agents[k].buffer.actions += temp_buffer.actions
                    rl_agents[k].buffer.action_mask += temp_buffer.action_mask
                    rl_agents[k].buffer.states += temp_buffer.states
                    rl_agents[k].buffer.vf_states += temp_buffer.vf_states
                    rl_agents[k].buffer.logprobs += temp_buffer.logprobs
                    rl_agents[k].buffer.state_values += temp_buffer.state_values
                    rl_agents[k].buffer.rewards += temp_buffer.rewards
                    rl_agents[k].buffer.frames += temp_buffer.frames
                    rl_agents[k].buffer.is_terminals += temp_buffer.is_terminals
                    rl_agents[k].buffer.returns += temp_buffer.returns
                    rl_agents[k].buffer.advantages += temp_buffer.advantages
                    if rl_agents[k].b_use_gae: rl_agents[k].buffer.td_errors += temp_buffer.td_errors

            if q < n_repeat_ep - 1:
                # Reset agents' rollout buffer and environments
                [ray.get(train_envs[j].reset_robf.remote()) for j in range(rl_agents[0].n_para)]
                [train_envs[j].init_new_episode.remote(rev_nseed=random.randint(1, 100000)) for j in range(rl_agents[0].n_para)]

        ret_ave, ret_std = np.mean(l_cum_rwd), np.std(l_cum_rwd)
        epl_ave, epl_std = np.mean(l_ep_len), np.std(l_ep_len)                
            
        # Save networks            
        if i > 0 and (i%n_save_model_ep == 0 or ret_max < ret_ave):
            for k in range(len(agt_names)):
                rl_agents[k].save(model_fname[k] + str(i) + '.pt')
                if ret_max < ret_ave: ret_max = ret_ave

        if i > 0: print(i, format(ret_ave, '.2f'), format(epl_ave, '.1f'), '('+format(time.perf_counter()-start_time, '.2f')+'s)')
        else: print(i, format(ret_ave, '.2f'), format(epl_ave, '.1f'), '('+format(time.perf_counter()-start_time, '.2f')+'s)')
                 
        start_time = time.perf_counter()

        for k in range(len(agt_names)):
            log_step_str = str(i)+','+str(ret_ave)+','+str(ret_std)+ ','+str(epl_ave)+ ','+str(epl_std)\
                            +','+str(rl_agents[k].loss_pi)+','+str(rl_agents[k].advantage)+','+str(rl_agents[k].loss_vf)+','+str(rl_agents[k].loss_s)\
                            +','+str(rl_agents[k].clip_frac)+','+str(rl_agents[k].approx_kl)+','+str(rl_agents[k].ret_actual)+','+str(rl_agents[k].v_estim)+','+str(rl_agents[k].expl_var)                
            if k == 0: 
                for p in l_cum_rwd: log_step_str += ',' + str(p)

            for p in l_ep_ret[k]: log_step_str += ',' + str(p)

            print('{}'.format(log_step_str), file=log_result[k])
            log_result[k].flush() 

            # Update actor network
            rl_agents[k].update(agt_names[k])           

            # Reset learners to each environment
            [ray.get(train_envs[j].reset_rl.remote(k, rl_agents[k])) for j in range(rl_agents[0].n_para)]
            
        [train_envs[j].init_new_episode.remote(rev_nseed=random.randint(1, 100000)) for j in range(rl_agents[0].n_para)]            

    [log_result[k].close() for k in range(len(agt_names))]
    ray.shutdown()


