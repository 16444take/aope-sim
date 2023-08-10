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

from sim_env.specs import *
from sim_env.util import *


@ray.remote
class AOPE_App:
    def __init__(self,
                 rl,
                 rlCritic,
                 n_seed = 0,
                 marl_type = 'cdsc',
                 order_type = 'LM',
                 l_rand_uniq_ship=[],
                 l_item_ones_place=[]):

        self.time = 0
        self.fps = 10                           # frames per second
        self.item_states = ['backlogged', "loading_conv", 'conv', 'pp1', 'rot', 'pp2', 'sorted']

        # Set random seed
        self.n_seed = n_seed                    # Random seed
        random.seed(self.n_seed)
        # Variation for agent's working speed
        self.sigma = 0.2                        # Standard deviation for Poisson distribution
        self.mu = 1.0                           # Mean

        self.all_trans_agt, self.all_reward, self.all_frame, self.all_state_value, self.all_is_terminal = [], [], [], [], []
        
        self.n_episode = 1
        self.old_obs, self.new_obs = [[]]*4, [[]]*4
        self.act, self.rwd = [0]*4, [0]*4
        self.obs_dim, self.act_dim = 0, 0
        self.n_buf = [0]*4
        self.ep_ret = [0.0]*4
        self.order_type = order_type
        self.l_rand_uniq_ship = l_rand_uniq_ship
        self.l_item_ones_place = l_item_ones_place

        self.rl = rl                                        # Set RL policy
        self.marl_type = marl_type
        if self.marl_type == 'cdsc': 
            self.rlCritic = rlCritic                        # Set RL critic
        self.b_do_act = [False]*4                           # Action flags for monitoring a time step when an agent has taken an action.

        if self.marl_type == 'illr': 
            self.t_act_latest = [0]*4                       # Time-step at the latest action (used for estimating immediate reward)

        # Read order data
        # 'ship_id': Shipping destination ID,  'product_id': Item type,  'orders': Number of ordered items 
        if self.order_type == 'LM':
            fpath = '.\\order_data\\LowMixedOrder.csv'
        elif self.order_type == 'HM':  
            fpath = '.\\order_data\\HighMixedOrder.csv'

        self.df = pd.read_csv(fpath, sep=',') 
        self.uniq_ship = self.df['ship_id'].unique().tolist()       # List of unique shipping destinations
        self.unship_id = deepcopy(self.uniq_ship)                   # List of unassigned shipping desitionations 
        self.uniq_type = self.df['product_id'].unique().tolist()    # List of unique item types
        self.n_ship = len(self.uniq_ship)                           # Number of total (shipping destinations) = (shipping boxes)
        self.n_pcs = self.df['orders'].sum()                        # Number of total items
        self.n_type = len(self.uniq_type)                           # Number of total item types
        self.n_order = self.df.shape[0]                             # Number of total picking orders

        # Set action mask to eliminate invalid actions
        self.mask = {0:[1]*self.rl[0].action_dim, 1:[0]*self.rl[1].action_dim, 2:[1]*self.rl[2].action_dim, 3:[0]*self.rl[3].action_dim}
        self.mask_shipid = {}
        self.locy_num = {}
        
        self.ships = {}                 # key:shipping destination、val:item dictionary
        self.ships_sta = {}             # key:shipping desitination、val:operational status
        self.orders = []                # List of picking orders
        self.stock_items = {}           # Dictionary of all stored items (key: , val:)
        self.in_proc_items = {}         # Dictionary of items in process (key:item_state, val:list[Item object])
        self.count_item = 0             # Number of in-process items (count up when items are added to self.in_proc_items)
        self.n_sorted_items = 0         # Number of completed items (count up when items are stored into shipping box on FR)

        for i in range(self.n_ship):
            item_pcs_sta = {}           # Key: item id, Val: [number of items, status]
            df_ship_per_id = self.df[self.df['ship_id'] == self.uniq_ship[i]] 
            for df_row in zip(df_ship_per_id['product_id'], df_ship_per_id['orders']):
               item_pcs_sta[df_row[0]] = [df_row[1], 'backlogged']
               self.orders.append(Order(order_id=len(self.orders), ship_id=self.uniq_ship[i], item_id=df_row[0], item_pcs=df_row[1], order_status='backlogged'))

            self.ships[self.uniq_ship[i]] = item_pcs_sta
            self.ships_sta[self.uniq_ship[i]] = 'backlogged'

        # :: ------------------------------------------------
        # :: Shipping box settings
        # :: ------------------------------------------------
        self.y0_sbox = 34                   # Y-coordinate of the topmost shipping box on layout (px)
        self.x_sbox = 197                   # X-coordinate of the boxes on layout (pix.)
        self.n_loc_sbox = 4                 # Number of shipping boxes that can be placed in FR
        self.space_sbox = 16                # Spacing of shipping boxes in FR (px)
        self.t_swap = 30 * self.fps         # Time to replace completed shipping boxes to empty new one (30 sec.)
        self.n_alloc_sbox = 0               # Number of allocated shipping boxes in FR
        self.n_accum_sbox = 0               # Total number of allocated shipping boxes in FR so far 
        self.alloc_sboxes = []              # List of allocated shipping boxes in operation

        # Status list of shipping box 
        self.sbox_states = ["loaded", "loading", "completed", "transporting"]

        # X/Y Position list of shipping boxes 
        self.pos_x_sbox = []    
        self.pos_y_sbox = []

        # X/Y Position list of stored items in a shipping box
        self.pos_x_item_in_sbox = []    
        self.pos_y_item_in_sbox = []

        # :: ------------------------------------------------
        # :: Carousel conveyor setting
        # :: ------------------------------------------------
        self.n_item_load = 28                       # Number of items that can be placed on carousel conveyor
        self.t_stay_in_same_loc = 2 * self.fps      # Rotational speed of carousel conveyor
        self.rotc = RotConveyor(conv_id = 0, 
                n_item_load = self.n_item_load, 
                t_stay_in_same_loc = self.t_stay_in_same_loc)

        # :: ------------------------------------------------
        # :: Parallel Conveyor (PC) setting
        # :: ------------------------------------------------
        # Set item locations
        # Store items to (n_loc) locations based on the first digit of item ID
        self.item_locations = {}    # key:Item ID  value:Distance to PR1
        self.n_loc = 6              # Number of item locations
        d_loc_ofs = 3600            # Distance between PR1 and nearest item-strage location
        d_loc_itvl = 3600           # Distance between negihbor item-strage locations
        self.l_item_locations = [d_loc_ofs + i*d_loc_itvl for i in range(self.n_loc)]
        for i in range(self.n_type):
            self.item_locations[self.uniq_type[i]] = self.l_item_locations[self.l_item_ones_place[i]]

        self.n_conv = 3                                                     # Number of parallel conveyors
        self.xend_conv = [82] * self.n_conv                                 # X-coordinate of conveyor where PR1 can pick up an item (px)       
        self.y_conv = [45, 61, 77]                                          # Y-coordinate of (three) conveyors
        self.put_interval = [4*self.fps, 4*self.fps, 4*self.fps]            # Time interval when items are put onto the conveyor [frame]
        self.item_intvl_conv = [6, 6, 6]                                    # Minimum distance between neighboring items (px)
        self.speed_ppf = [12, 12, 12]                                       # Conveyor speed per frame (px/frm)
        self.intvl_loading = [10*self.fps, 10*self.fps, 10*self.fps]        # Time interval required to put different type items onto conveyor [frame]  

        for i in range(self.n_conv): 
            assert self.speed_ppf[i]%self.item_intvl_conv[i] == 0 , 'Conveyor speed should be a multiple of item interval'

        # Get the number of possible placement locations on the conveyor based on the length and speed.
        self.num_locations_conv = [int(max(list(self.item_locations.values())) / self.item_intvl_conv[i]) for i in range(self.n_conv)]

        self.convs = []
        [self.convs.append(Conveyor(conv_id=i, conv_xend=self.xend_conv[i], conv_y=self.y_conv[i], put_interval=self.put_interval[i],
                                    item_intvl_conv=self.item_intvl_conv[i], speed_ppf=self.speed_ppf[i], num_locations_conv=self.num_locations_conv[i])) 
         for i in range(self.n_conv)]

        for i in range(self.n_conv):
            entry_x_conv = {} # key: Item ID,  val: List of X-coordinates of locations on the conveyor where items are to be loaded
            for k, v in self.item_locations.items():
                min_xloc = []            
                min_j, min_dist = 0, 1e4
                for j in range(len(self.convs[i].locx_conv)):
                    diff_x = abs(abs(self.convs[i].locx_conv[0] - self.convs[i].locx_conv[j]) - v)
                    if diff_x < min_dist: min_j, min_dist = j, diff_x
                min_xloc.append(self.convs[i].locx_conv[min_j])
                entry_x_conv[k] = self.convs[i].locx_conv[min_j]            
            self.convs[i].x_entries = entry_x_conv

        # Set the item placement location on the carousel conveyor corresponding to each conveyor
        for i in range (self.n_conv): self.rotc.SetPlaceLocations(self.convs[i])

        # :: ------------------------------------------------
        # :: Picking Robot 1 (PR1) setting
        # :: ------------------------------------------------
        self.t_p1_pick   = 2 * self.fps          # Time required to pick up an item from PC
        self.t_p1_place  = 2 * self.fps          # Time required to place an item onto carousel conveyor 
        self.t_p1_rot_pi = 3 * self.fps          # Time required to rotate 180deg between PC and carousel conveyor
        self.t_p1_move   = 5 * self.fps          # Time required to move between PCs
        self.t_p1_start  = 20 * self.fps         # Work start time
        self.n_p1_pick   = 1                     # Number of items to pick at a time
        # XY-coordinates of PR1 [px]
        self.x_p1_pos =  [88 for _ in range(self.n_conv)]
        self.y_p1_pos =  [40+i*16 for i in range(self.n_conv)]

        self.probo1 = PickRobo(robo_id = 1, n_pick = self.n_p1_pick, t_pick = self.t_p1_pick, t_place = self.t_p1_place,
                              t_rot_pi = self.t_p1_rot_pi, t_move = self.t_p1_move, x_pos = self.x_p1_pos, y_pos = self.y_p1_pos,
                              t_start = self.t_p1_start)

        # :: ------------------------------------------------
        # :: Picking Robot 2 (PR2) setting
        # :: ------------------------------------------------
        self.t_p2_pick   = 1 * self.fps        # Time required to pick up an item from carousel conveyor 
        self.t_p2_place  = 2 * self.fps        # Time required to place an item to shipping box
        self.t_p2_rot_pi = 2 * self.fps        # Time required to rotate 180deg between carousel conveyor and FR
        self.t_p2_move   = 3 * self.fps        # Time required to move between shipping boxes
        self.t_p2_start  = 0                   # Work start time
        self.n_p2_pick   = 1                   # Number of items to pick at a time
        # XY-coordinates of PR2 [px]
        self.x_p2_pos =  [176 for _ in range(self.n_loc_sbox)]
        self.y_p2_pos =  [34+i*16 for i in range(self.n_loc_sbox)]

        self.probo2 = PickRobo(robo_id=2, n_pick=self.n_p2_pick, t_pick=self.t_p2_pick, t_place=self.t_p2_place,
                              t_rot_pi =self. t_p2_rot_pi, t_move=self.t_p2_move, x_pos=self.x_p2_pos, y_pos=self.y_p2_pos,
                              t_start = self.t_p2_start)

        # Set the item pick position on carousel conveyor corresponding to each shipping box position
        self.rotc.SetPickLocations(self.probo2)

        # Place shipping boxes on FR at the simulation start and allocates shipping destinations
        for i in range(self.n_loc_sbox):
            if self.rl[0] != '': # Exist RL policy
                if self.b_do_act[0]: # Previously made decision             
                    self.all_is_terminal.append(False), self.rl[0].buffer.is_terminals.append(False)
                    self.n_buf[0] += 1
                    self.b_do_act[0] = False
                    self.rwd[0] = self.calc_immediate_reward(0)
                    self.rl[0].buffer.rewards.append(0.0)
                    self.ep_ret[0] += self.rwd[0]
                 
                obs_list = [] # policy states
                unsorted_item_type, unsorted_item_pcs = [], 0
                for j in range(len(self.alloc_sboxes)):
                    unsorted_items = self.alloc_sboxes[j].unsorted_items
                    for k, v in unsorted_items.items():
                        unsorted_item_pcs += v
                        if k not in unsorted_item_type: unsorted_item_type.append(k)

                # Get item information included in unshipped shipping destinations
                unships, unship_item_type, unship_item_pcs = 0, 0, 0
                for key, val in self.ships.items():
                    if key not in self.unship_id: continue
                    unships += 1
                    for key_itm, val_itm in val.items():
                        unship_item_pcs += val_itm[0]
                        unship_item_type += 1

                # Get item information included in unshipped shipping destinations
                stored_items =[]
                if 'backlogged' in self.in_proc_items: stored_items = [x.type for x in self.in_proc_items['backlogged']]
                obs_list.extend([len(unsorted_item_type), unsorted_item_pcs, len(set(stored_items)), len(stored_items), 
                                 unships/5, unship_item_type/20, unship_item_pcs/20])   
                
                self.old_obs[0] = obs_list

                if self.marl_type == 'cdsc' or self.marl_type == 'cdic': 
                    obs_vf_list = self.get_all_states(0, -1) # critic states
                
                if self.marl_type == 'cdsc':
                    self.act[0] = self.rl[0].select_action(self.old_obs[0]+self.mask[0]+[(self.time+i)/5e4], self.time+i, self.mask[0])
                    state_value = self.rlCritic.estimate_vf(obs_vf_list)
                elif self.marl_type == 'cdic':
                    self.act[0], state_value = self.rl[0].select_action(self.old_obs[0]+self.mask[0]+[(self.time+i)/5e4], obs_vf_list, self.time+i, self.mask[0])                
                else: # ILLR or ILGR
                    self.act[0] = self.rl[0].select_action(self.old_obs[0] + self.mask[0] + [(self.time+i)/5e4], self.time+i, self.mask[0])
                
                self.mask[0][self.act[0]] = 0 # Set action mask for currently selected shipping box 
                self.b_do_act[0] = True
                if self.marl_type == 'illr': self.t_act_latest[0] = self.time + i

                self.all_trans_agt.append(0), self.all_frame.append(self.time+i)
                if self.marl_type == 'cdsc' or self.marl_type == 'cdic': 
                    self.all_state_value.append(state_value)

            if i == 0:
                ship_id = self.uniq_ship[self.act[0]]
                self.ships_sta[ship_id] = 'allocated'
            else:
                ship_id = self.uniq_ship[self.act[0]]
                self.ships_sta[ship_id] = 'allocated'

            self.pos_x_sbox.append(self.x_sbox)
            self.pos_y_sbox.append(self.y0_sbox + self.space_sbox * i)
            self.locy_num[self.y0_sbox + self.space_sbox * i] = i
            # Add items contained in the selected shipping box to the in-process dictionary
            d_items_in_sbox = self.ships[ship_id]
            d_items_pcs_in_sbox = {} 
            for k, v in d_items_in_sbox.items():
                 n_added_item = regist_item_in_proc(n_item=v[0], status=self.item_states[0], itm_type=k,
                                                   count_item=self.count_item, d_item=self.in_proc_items, in_time=self.time)
                 self.count_item += n_added_item  
                 d_items_pcs_in_sbox[k] = v[0]

            self.alloc_sboxes.append(Sbox(box_id = self.n_accum_sbox, ship_id = ship_id, 
                                          box_t_in = self.time, box_status = self.sbox_states[0],  
                                          box_x = self.pos_x_sbox[i], box_y = self.pos_y_sbox[i],
                                          unsorted_items = d_items_pcs_in_sbox, t_swap = self.t_swap))
            self.unship_id.remove(ship_id)
            self.n_alloc_sbox += 1          
            self.n_accum_sbox += 1 

        # Advance four (= number of FR decisions) time steps 
        self.time += 4 


    def __del__(self): pass

    # ==================================================================================================
    # Run order picking episode
    def run_episode(self, t_max):
        for i in range(int(t_max)):
            b_end = self.update()
            if b_end: break
        return self.rl[0].ep_len

    # ==================================================================================================
    # Get all agents' states
    def get_all_states(self, agt_id, conv_id):
        # -------------
        # Common
        # -------------
        time_mod_rotc = self.time % int(self.rotc.n_item_load*self.rotc.t_stay_in_same_loc)
        l_states = [conv_id]
        l_states.append(agt_id)
        l_states.append(self.time/5e4)
        for i in range(4):l_states.extend(self.mask[i]) # action masks 

        # -------------
        # FR states
        # -------------
        unsorted_item_type, unsorted_item_pcs = [], 0
        for j in range(len(self.alloc_sboxes)):
            unsorted_items = self.alloc_sboxes[j].unsorted_items
            for k, v in unsorted_items.items():
               unsorted_item_pcs += v
               if k not in unsorted_item_type: unsorted_item_type.append(k)
        unships, unship_item_type, unship_item_pcs = 0, 0, 0
        for key, val in self.ships.items():
            if key not in self.unship_id: continue
            unships += 1
            for key_itm, val_itm in val.items():
                unship_item_pcs += val_itm[0]
                unship_item_type += 1
        stored_items =[]
        if 'backlogged' in self.in_proc_items: stored_items = [x.type for x in self.in_proc_items['backlogged']]
        l_states.extend([len(unsorted_item_type), unsorted_item_pcs, len(set(stored_items)), len(stored_items), 
                    unships/5, unship_item_type/20, unship_item_pcs/20]) 

        # -------------
        # PC states
        # -------------
        loading_items = [] 
        for i in range(self.n_conv):      
            if self.convs[i].loading_item == '': continue 
            loading_items += [self.convs[i].loading_item]

        if 'backlogged' not in self.in_proc_items: 
            l_states.extend([0]*27)
        else:
            l_items = self.in_proc_items['backlogged']                     
            if len((l_items)) > 0:
                l_candi_items = []
                d_items = {}
                for j in range(len(l_items)):
                    if l_items[j].type in loading_items: continue
                    l_candi_items += [l_items[j]]
                    if l_items[j].type not in d_items.keys(): d_items[l_items[j].type] = 1
                    else: d_items[l_items[j].type] += 1
                for j in range(self.n_conv):
                    d_item_conv = self.convs[j].x_loaded_items
                    if len(d_item_conv) == 0: 
                        l_states.extend([0, 0, 0])
                    else:
                        l_loaded_items = list(d_item_conv.keys())
                        l_states.extend([(min(l_loaded_items)+2.17e4)/2.17e4*20-10, (max(l_loaded_items)+2.17e4)/2.17e4*20-10, len(d_item_conv)])
                d_candi_pcs, d_candi_type = {}, {}
                l_uniq_candi_type = []
                for cand_item in l_candi_items:
                    loc_candi = self.item_locations[cand_item.type]
                    if loc_candi not in d_candi_pcs:
                        d_candi_pcs[loc_candi] = 1
                        d_candi_type[loc_candi] = 1
                        l_uniq_candi_type.append(cand_item.type)
                    else:
                        d_candi_pcs[loc_candi] += 1
                        if cand_item.type not in l_uniq_candi_type: 
                            l_uniq_candi_type.append(cand_item.type)
                            d_candi_type[loc_candi] = 1
                for j in range(self.n_loc):
                    if self.l_item_locations[j] not in d_candi_pcs:
                        l_states.extend([0.0, 0.0])
                    else:
                        l_states.extend([d_candi_pcs[self.l_item_locations[j]], d_candi_type[self.l_item_locations[j]]])
                for j in range(self.n_conv):
                    if j == i or self.convs[j].loading_item == '':
                        l_states.extend([0.0, 0.0])
                    else:
                        l_loading_items = self.in_proc_items['loading_conv']
                        count_loading_items = 0
                        for itm in l_loading_items:
                            if itm.type == self.convs[j].loading_item: count_loading_items+=1
                        l_states.extend([self.l_item_locations.index(self.item_locations[l_loading_items[0].type]), count_loading_items])
            else:
                l_states.extend([0]*27)

        # -------------
        # PR1 states
        # -------------
        l_states.extend([self.probo1.ny, self.probo1.dir])
        if 'rotc' in self.in_proc_items: l_states.extend([len(self.in_proc_items['rotc'])])
        else:l_states.extend([0])

        for i in range(len(self.convs)):
            d_item_conv = self.convs[i].x_loaded_items
            d_item_conv = dict(sorted(self.convs[i].x_loaded_items.items(), key=itemgetter(0), reverse=True))	
            if len(d_item_conv) == 0:
                l_states.extend([-20, 0, -20, 0])
                continue
            pos_item, head_item_type = next(iter(d_item_conv)), d_item_conv[next(iter(d_item_conv))] 
            pos_item_last = min(d_item_conv)
            n_head_items = 0
            for k, v in d_item_conv.items():
                if v == head_item_type: n_head_items += 1
                else: break	
            n_tot_items = len(d_item_conv.items()) - n_head_items
            l_states.extend([(pos_item+2.17e4)/2.17e4*20-10, n_head_items, (pos_item_last+2.17e4)/2.17e4*20-10, n_tot_items])

        loading_items = [] 
        for i in range(len(self.convs)):
            if self.convs[i].loading_item != '':
                l_loading_items = self.in_proc_items['loading_conv']
                count_loading_items = 0
                for itm in l_loading_items:
                    if itm.type == self.convs[i].loading_item: count_loading_items += 1
                l_states.extend([(self.convs[i].x_lock_for_loading+2.17e4)/2.17e4*20-10, count_loading_items])
                loading_items += [self.convs[i].loading_item]
            else: l_states.extend([-10, 0])

        # -------------
        # PR2 states
        # -------------
        l_states.extend([(self.n_sorted_items-100)/20, self.probo2.ny*2-4, self.probo2.dir*2-4])
        if len(self.alloc_sboxes) < self.n_loc_sbox:
            for i in range(self.n_loc_sbox):
                l_states.extend([-1]*self.n_item_load)
                l_states.extend([0])
        else:
            l_cand_sbox = []
            if 'rotc' in self.in_proc_items.keys():
                items_rotc = self.in_proc_items['rotc']
                set_item_type_rotc_uniq = set([item_info.type for item_info in items_rotc]) 
                for i in range(len(self.alloc_sboxes)):
                    if self.alloc_sboxes[i].b_completed: continue
                    set_sbox_unsorted_items = set(self.alloc_sboxes[i].unsorted_items) 
                    if len(set_item_type_rotc_uniq & set_sbox_unsorted_items): l_cand_sbox.append(i)
                l_loaded_items_rotc = self.rotc.GetLoadedItemArray(time_mod_rotc)
                for i in range(len(self.alloc_sboxes)):
                    if i not in l_cand_sbox:
                        l_states.extend([-1]*self.n_item_load)
                    else:
                        l_unsorted_items =list(self.alloc_sboxes[i].unsorted_items.keys())
                        for rotc_item in l_loaded_items_rotc:
                            if rotc_item == '': l_states.append(-5)
                            elif rotc_item in l_unsorted_items: l_states.append(5)
                            else: l_states.append(-5)
                    l_states.append(sum(list(self.alloc_sboxes[i].unsorted_items.values())))
            else:
                for i in range(self.n_loc_sbox):
                    l_states.extend([-5]*self.n_item_load)
                    l_states.extend([0])

        return l_states

    # ==================================================================================================
    # Get rollouts for updating policies
    def get_history(self, learner_id):
        if self.marl_type == 'illr': 
            return self.rl[learner_id].buffer, self.rl[learner_id].cum_rwd, self.rl[learner_id].ep_len, self.ep_ret[learner_id]
        else: 
            return self.rl[learner_id].buffer, self.rl[learner_id].cum_rwd, self.rl[learner_id].ep_len

    # ==================================================================================================
    # Get rollouts for updating critic
    def get_critic_history(self):
        return self.rlCritic.buffer

    # ==================================================================================================
    # Estimate immediate reward
    def calc_immediate_reward(self, agt_id):
        if self.marl_type == 'illr':
            return (self.t_act_latest[agt_id] - self.time)/10
        else:            
            return 0.0 # No immediate reward

    # ==================================================================================================
    # Estimate terminal reward
    def calc_terminal_reward(self):
        factor = 0.2
        if self.order_type == 'LM': 
            t_diff = -self.time/self.fps/1e3 + 6.6
            if t_diff > 0: return  (factor * 10 * abs(t_diff)**4.0 - factor * 10)/1 
            else: return  (-10 * factor * abs(t_diff)**4.0 - factor * 10)/1 
        elif self.order_type == 'HM': 
            t_diff = -self.time/self.fps/1e3 + 6.4
            if t_diff > 0: return  factor * 10 * abs(t_diff)**4.0 - factor * 10 
            else: return  -10 * factor * abs(t_diff)**4.0 - factor * 10

    # ==================================================================================================
    # Reset RL policy
    def reset_rl(self, agent_id, rl):
        self.rl[agent_id] = rl
        return(len(self.rl[agent_id].buffer.returns))

    # ==================================================================================================
    # Reset RL critic
    def reset_rl_critic(self, rlCritic):
        self.rlCritic = rlCritic
        return(len(self.rlCritic.buffer.returns))

    # ==================================================================================================
    # Reset RL rollout buffer
    def reset_robf(self):
        for i in range(4): self.rl[i].buffer.clear(self.rl[i].b_use_gae)
        if self.marl_type == 'cdsc': self.rlCritic.buffer.clear(self.rlCritic.b_use_gae)        
        return(len(self.rl[0].buffer.returns))

    # ==================================================================================================
    #  Initialize simulation parameters to get new rollout
    def init_new_episode(self, rev_nseed=0):
        self.time = 0        
        random.seed(rev_nseed)
        self.n_episode += 1
        self.b_do_act = [False]*4 
        self.n_buf = [0]*4
        self.ep_ret = [0.0]*4
        self.unship_id = deepcopy(self.uniq_ship)     
        self.ships = {}        
        self.ships_sta = {}    
        self.orders = []      
        self.stock_items = {}   
        self.in_proc_items = {} 
        self.count_item = 0   
        self.n_sorted_items = 0 
        self.rwd = [0.0]*4
        self.all_trans_agt, self.all_reward, self.all_frame, self.all_state_value, self.all_is_terminal = [], [], [], [], []
        self.mask = {0:[1]*self.rl[0].action_dim, 1:[0]*self.rl[1].action_dim, 2:[1]*self.rl[2].action_dim, 3:[0]*self.rl[3].action_dim}
        if self.marl_type == 'illr': 
            self.t_act_latest = [0]*4 

        for i in range(self.n_ship):
            item_pcs_sta = {} 
            df_ship_per_id = self.df[self.df['ship_id'] == self.uniq_ship[i]] 
            for df_row in zip(df_ship_per_id['product_id'], df_ship_per_id['orders']):
               item_pcs_sta[df_row[0]] = [df_row[1], 'backlogged']
               self.orders.append(Order(order_id=len(self.orders), 
                                        ship_id=self.uniq_ship[i], 
                                        item_id=df_row[0], 
                                        item_pcs=df_row[1], 
                                        order_status='backlogged'))

            self.ships[self.uniq_ship[i]] = item_pcs_sta
            self.ships_sta[self.uniq_ship[i]] = 'backlogged'

        self.n_alloc_sbox = 0   
        self.n_accum_sbox = 0   
        self.alloc_sboxes = []

        self.rotc = RotConveyor(conv_id=0, n_item_load=self.n_item_load, t_stay_in_same_loc=self.t_stay_in_same_loc)

        self.convs = []
        [self.convs.append(Conveyor(conv_id=i, conv_xend=self.xend_conv[i], conv_y=self.y_conv[i], put_interval=self.put_interval[i],
                                    item_intvl_conv=self.item_intvl_conv[i], speed_ppf=self.speed_ppf[i], num_locations_conv=self.num_locations_conv[i]))
         for i in range(self.n_conv)]

        for i in range(self.n_conv):
            entry_x_conv = {} 
            for k, v in self.item_locations.items():
                min_xloc = []            
                min_j, min_dist = 0, 1e4
                for j in range(len(self.convs[i].locx_conv)):
                    diff_x = abs(abs(self.convs[i].locx_conv[0] - self.convs[i].locx_conv[j]) - v)
                    if diff_x < min_dist: min_j, min_dist = j, diff_x
                min_xloc.append(self.convs[i].locx_conv[min_j])
                entry_x_conv[k] = self.convs[i].locx_conv[min_j]
            
            self.convs[i].x_entries = entry_x_conv

        for i in range (self.n_conv): self.rotc.SetPlaceLocations(self.convs[i])

        self.probo2 = PickRobo(robo_id = 2, n_pick = self.n_p2_pick, t_pick = self.t_p2_pick, t_place = self.t_p2_place,
                              t_rot_pi = self. t_p2_rot_pi, t_move = self.t_p2_move, x_pos = self.x_p2_pos, y_pos = self.y_p2_pos,
                              t_start = self.t_p2_start)
        self.rotc.SetPickLocations(self.probo2)

        self.probo1 = PickRobo(robo_id=1, n_pick=self.n_p1_pick, t_pick=self.t_p1_pick, t_place=self.t_p1_place,
                              t_rot_pi = self.t_p1_rot_pi, t_move = self.t_p1_move, x_pos = self.x_p1_pos, y_pos = self.y_p1_pos,
                              t_start = self.t_p1_start)

        for i in range(self.n_loc_sbox):
            if self.rl[0] != '': 
                if self.b_do_act[0]:
                    self.all_is_terminal.append(False), self.rl[0].buffer.is_terminals.append(False)
                    self.n_buf[0] += 1
                    self.b_do_act[0] = False

                    self.rwd[0] = self.calc_immediate_reward(0)
                    self.rl[0].buffer.rewards.append(self.rwd[0])
                    self.ep_ret[0] += self.rwd[0]

                obs_list = []
                unsorted_item_type, unsorted_item_pcs = [], 0
                for j in range(len(self.alloc_sboxes)):
                    unsorted_items = self.alloc_sboxes[j].unsorted_items
                    for k, v in unsorted_items.items():
                        unsorted_item_pcs += v
                        if k not in unsorted_item_type: unsorted_item_type.append(k)

                unships, unship_item_type, unship_item_pcs = 0, 0, 0
                for key, val in self.ships.items():
                    if key not in self.unship_id: continue
                    unships += 1
                    for key_itm, val_itm in val.items():
                        unship_item_pcs += val_itm[0]
                        unship_item_type += 1

                stored_items =[]
                if 'backlogged' in self.in_proc_items: stored_items = [x.type for x in self.in_proc_items['backlogged']]

                obs_list.extend([len(unsorted_item_type), unsorted_item_pcs, len(set(stored_items)), len(stored_items), 
                                 unships/5, unship_item_type/20, unship_item_pcs/20])  

                self.old_obs[0] = obs_list

                if self.marl_type == 'cdsc' or self.marl_type == 'cdic': 
                    obs_vf_list = self.get_all_states(0, -1)
                
                if self.marl_type == 'cdsc':
                    self.act[0] = self.rl[0].select_action(self.old_obs[0]+self.mask[0]+[(self.time+i)/5e4], self.time+i, self.mask[0])
                    state_value = self.rlCritic.estimate_vf(obs_vf_list)
                elif self.marl_type == 'cdic':
                    self.act[0], state_value = self.rl[0].select_action(self.old_obs[0]+self.mask[0]+[(self.time+i)/5e4], obs_vf_list, self.time, self.mask[0])
                else:
                    self.act[0] = self.rl[0].select_action(self.old_obs[0] + self.mask[0] + [(self.time+i)/5e4], self.time+i, self.mask[0])

                self.mask[0][self.act[0]] = 0 
                self.b_do_act[0] = True
                if self.marl_type == 'illr': self.t_act_latest[0] = self.time + i

                self.all_trans_agt.append(0), self.all_frame.append(self.time+i)
                if self.marl_type == 'cdsc' or self.marl_type == 'cdic': 
                    self.all_state_value.append(state_value)

            if i == 0:
                ship_id = self.uniq_ship[self.act[0]]
                self.ships_sta[ship_id] = 'allocated'
            else:
                ship_id = self.uniq_ship[self.act[0]]
                self.ships_sta[ship_id] = 'allocated'

            d_items_in_sbox = self.ships[ship_id]
            d_items_pcs_in_sbox = {} 
            for k, v in d_items_in_sbox.items():
                 n_added_item = regist_item_in_proc(n_item=v[0], status=self.item_states[0], itm_type=k,
                                                   count_item=self.count_item, d_item=self.in_proc_items, in_time=self.time)
                 self.count_item += n_added_item
                 d_items_pcs_in_sbox[k] = v[0]

            self.alloc_sboxes.append(Sbox(box_id = self.n_accum_sbox, ship_id = ship_id, 
                                          box_t_in = self.time, box_status = self.sbox_states[0],  
                                          box_x = self.pos_x_sbox[i], box_y = self.pos_y_sbox[i],
                                          unsorted_items = d_items_pcs_in_sbox, t_swap = self.t_swap))
            self.unship_id.remove(ship_id)
            self.n_alloc_sbox += 1          
            self.n_accum_sbox += 1 

        self.time += 4 


    # ==================================================================================================
    # advance one time step
    def update(self):
        # Update system status from down stream process

        # Convert to rotating time according to the cycle of carousel conveyor
        time_mod_rotc = self.time % int(self.rotc.n_item_load*self.rotc.t_stay_in_same_loc)      

        # :: ----------------------------------------------------------------
        # :: FR
        # :: ----------------------------------------------------------------
        n_end_sbox = 0
        for i in range(len(self.alloc_sboxes)):
            if self.alloc_sboxes[i].t_wait_swap > 0: 
                self.alloc_sboxes[i].x += self.alloc_sboxes[i].speed
                self.alloc_sboxes[i].t_wait_swap -= 1
                if self.alloc_sboxes[i].t_wait_swap == int(self.alloc_sboxes[i].t_swap/2):
                    # Send empty box to FR
                    self.alloc_sboxes[i].speed = -1*self.alloc_sboxes[i].speed
                    self.alloc_sboxes[i].EmptySbox()
            elif self.alloc_sboxes[i].status == 'completed':
                self.alloc_sboxes[i].x = self.pos_x_sbox[i]
                if self.n_ship == self.n_accum_sbox: 
                    n_end_sbox += 1
                    continue

                # If unallocated shipping destinations remain, allocate new destination to the empty shipping box.
                if self.rl[0] != '':

                    if self.rl[0].b_na_one and sum(self.mask[0]) == 1: 
                        self.act[0] = self.mask[0].index(1)
                    else:
                        if self.b_do_act[0]:
                            self.all_is_terminal.append(False), self.rl[0].buffer.is_terminals.append(False)
                            self.n_buf[0] += 1
                            self.b_do_act[0] = False

                            self.rwd[0] = self.calc_immediate_reward(0)
                            self.rl[0].buffer.rewards.append(self.rwd[0])
                            self.ep_ret[0] += self.rwd[0]

                        # Get RL states 
                        obs_list = []
                        unsorted_item_type, unsorted_item_pcs = [], 0
                        for j in range(len(self.alloc_sboxes)):
                            unsorted_items = self.alloc_sboxes[j].unsorted_items
                            for k, v in unsorted_items.items():
                                unsorted_item_pcs += v
                                if k not in unsorted_item_type: unsorted_item_type.append(k)
                    
                        unships, unship_item_type, unship_item_pcs = 0, 0, 0
                        for key, val in self.ships.items():
                            if key not in self.unship_id: continue
                            unships += 1
                            for key_itm, val_itm in val.items():
                                unship_item_pcs += val_itm[0]
                                unship_item_type += 1

                        stored_items = []
                        if 'backlogged' in self.in_proc_items: stored_items = [x.type for x in self.in_proc_items['backlogged']]

                        obs_list.extend([len(unsorted_item_type), unsorted_item_pcs, len(set(stored_items)), len(stored_items), 
                                     unships/5, unship_item_type/20, unship_item_pcs/20]) 

                        self.old_obs[0] = obs_list

                        if self.marl_type == 'cdsc' or self.marl_type == 'cdic':
                            obs_vf_list = self.get_all_states(0, -1)
                        
                        if self.marl_type == 'cdsc':
                            self.act[0] = self.rl[0].select_action(self.old_obs[0]+self.mask[0]+[self.time/5e4], self.time, self.mask[0])
                            state_value = self.rlCritic.estimate_vf(obs_vf_list)
                        elif self.marl_type == 'cdic':
                            self.act[0], state_value = self.rl[0].select_action(self.old_obs[0]+self.mask[0]+[self.time/5e4], obs_vf_list, self.time, self.mask[0])
                        else:
                            self.act[0] = self.rl[0].select_action(self.old_obs[0] + self.mask[0] + [self.time/5e4], self.time, self.mask[0])
                        
                        self.b_do_act[0] = True

                        self.all_trans_agt.append(0), self.all_frame.append(self.time)
                        if self.marl_type == 'cdsc' or self.marl_type == 'cdic': 
                            self.all_state_value.append(state_value)

                    self.mask[0][self.act[0]] = 0 # Set action mask for currently selected shipping box                   

                ship_id = self.uniq_ship[self.act[0]]
                self.ships_sta[ship_id] = 'allocated'
                
                # Add items contained in the selected shipping box to the in-process dictionary
                d_items_in_sbox = self.ships[ship_id]
                d_items_pcs_in_sbox = {}
                for k, v in d_items_in_sbox.items():
                     n_added_item = regist_item_in_proc(n_item=v[0], status=self.item_states[0], itm_type=k,
                                                   count_item=self.count_item, d_item=self.in_proc_items, in_time=self.time)
                     self.count_item += n_added_item
                     d_items_pcs_in_sbox[k] = v[0]

                self.alloc_sboxes[i] = Sbox(box_id = self.n_accum_sbox, ship_id = ship_id, 
                                              box_t_in = self.time, box_status = self.sbox_states[0],  
                                              box_x = self.pos_x_sbox[i], box_y = self.pos_y_sbox[i],
                                              unsorted_items = d_items_pcs_in_sbox, t_swap = self.t_swap)
                self.unship_id.remove(ship_id)
                self.n_alloc_sbox += 1          
                self.n_accum_sbox += 1  

        # :: ------------------------------------
        # :: Simulation termination process 
        # :: ------------------------------------
        if n_end_sbox == self.n_loc_sbox:
            self.time_end = time.perf_counter()
                                
            if all(self.b_do_act):
                for i in range(4):
                    # calculate terminal reward
                    if self.marl_type == 'illr': self.rwd[i] = self.calc_immediate_reward(i)
                    else: self.rwd[i] = self.calc_terminal_reward()
                    self.b_do_act[i] = False
                    self.n_buf[i] +=1                
                
            self.all_is_terminal.append(True)
            self.all_is_terminal = [False] * len(self.all_trans_agt)
            self.all_is_terminal[-1] = True
            self.all_reward = [0] * len(self.all_is_terminal)
            self.all_reward[-1] = self.calc_terminal_reward()

            if self.marl_type == 'cdsc': 
                # Asynchronous generalized advantage estimation with single centralized critic 
                returns, td_errors, advantages = [], [], []
                cum_discounted_reward, last_gae = 0, 0
                frame_prev, state_value_prev = self.all_frame[-1], self.all_state_value[-1]
                for reward, is_terminal, frame, state_value in zip(reversed(self.all_reward), reversed(self.all_is_terminal), 
                                                           reversed(self.all_frame), reversed(self.all_state_value)):

                    pow_index = (frame_prev-frame)/self.rl[0].gamma_scale
                    cum_discounted_reward = reward + (self.rl[0].gamma**(pow_index)) * cum_discounted_reward
                    
                    if is_terminal: td_error = cum_discounted_reward - state_value
                    else: td_error = reward + self.rl[0].gamma**(pow_index)*state_value_prev - state_value
                
                    returns.insert(0, cum_discounted_reward)
                    td_errors.insert(0, td_error)
                    last_gae = td_error + ((self.rl[0].gamma*self.rl[0].gae_lambda)**(pow_index) * last_gae)
                    advantages.insert(0, last_gae)
                    frame_prev, state_value_prev = frame, state_value

                self.rlCritic.buffer.returns.extend(returns)

            for i in range(4):
                self.ep_ret[i] += self.rwd[i]
                self.rl[i].cum_rwd = self.calc_terminal_reward()
                self.rl[i].ep_len = self.time
                self.rl[i].buffer.rewards.append(self.rwd[i])
                self.rl[i].buffer.is_terminals.append(True)
                if self.marl_type == 'cdsc': self.rl[i].calc_return(i, self.all_trans_agt, returns, td_errors, advantages, self.all_state_value)
                else: self.rl[i].calc_return()

            return True

        # :: ----------------------------------------------------------------
        # :: PR2
        # :: ----------------------------------------------------------------
        if not self.probo2.b_start: 
            # Count down to start next work
            if self.probo2.t_start > 0: self.probo2.t_start -= 1
            if self.probo2.t_start == 0 and len(self.rotc.pos_loaded_rotc) >= 1: self.probo2.b_start = True
        else:
            # Count down wating times
            if self.probo2.t_wait_move > 0 or self.probo2.t_wait_rot > 0:
                # Count down moving time and rotating time
                if self.probo2.t_wait_move > 0:
                    self.probo2.t_wait_move -= 1
                    self.probo2.y = self.probo2.y+self.probo2.vel                    
                    if self.probo2.t_wait_move == 0: # PR2 arrives at a shipping box
                        self.probo2.status.remove('moving')                 # Remove state flag "moving"
                        self.probo2.y = self.probo2.y_pos[self.probo2.ny]   # Set precise position
                        self.probo2.vel = 0                                 # Set velocity to zero
                if self.probo2.t_wait_rot > 0:
                    self.probo2.t_wait_rot -= 1
                    if self.probo2.t_wait_rot == 0: # Rotation completed
                        if 'rotating1' in self.probo2.status: # Completed to turning to placing side
                            self.probo2.status.remove('rotating1')    
                            self.probo2.change_dir(3) # Change PR2 direction to placing side
                        else: # Completed to turning to picking side
                            self.probo2.status.remove('rotating2')       
                            self.probo2.change_dir(2) # Change PR2 direction to picking side

                if self.probo2.t_wait_move == 0 and self.probo2.t_wait_rot == 0:
                    # Confirm availability to pick or place items
                    if self.probo2.dir == 3: self.probo2.status.append('can_place')
                    elif self.probo2.sort_sbox == '': self.probo2.status.append('free')
                    else: self.probo2.status.append('can_pick')
            elif self.probo2.t_wait_pick > 0:
                # Count down picking time               
                self.probo2.t_wait_pick -= 1
            elif self.probo2.t_wait_place > 0:
                # Count down placing time
                self.probo2.t_wait_place -= 1

            # Assign new picking task to PR2 if it's free
            if 'free' in self.probo2.status:    
                item_type, sbox_id, ship_id = '', '', ''
                if 'rotc' in self.in_proc_items.keys():
                    items_rotc = self.in_proc_items['rotc']
                    if len(items_rotc) == 0: pass

                    # Check how many boxes items can be sorted into
                    l_cand_sbox = []
                    set_item_type_rotc_uniq = set([item_info.type for item_info in items_rotc]) # Get unique item types on carousel conveyor
                    for i in range(len(self.alloc_sboxes)):
                        if self.alloc_sboxes[i].b_completed: continue
                        set_sbox_unsorted_items = set(self.alloc_sboxes[i].unsorted_items) # Get unique item types required for i-th shipping box
                        if len(set_item_type_rotc_uniq & set_sbox_unsorted_items): l_cand_sbox.append(i)

                    if self.rl[3].b_na_one and len(l_cand_sbox) == 1: # Exist only one available sipping box for sorting
                        sbox_id, ship_id = l_cand_sbox[0], self.alloc_sboxes[0].ship_id
                    elif len(l_cand_sbox) > 0: #  Exist more than two available boxes
                        if self.rl[3] != '':# RL case
                            if self.b_do_act[3]:                                 
                                self.all_is_terminal.append(False), self.rl[3].buffer.is_terminals.append(False)
                                self.n_buf[3] += 1
                                self.b_do_act[3] = False
                                self.rwd[3] = self.calc_immediate_reward(3)
                                self.rl[3].buffer.rewards.append(self.rwd[3])
                                self.ep_ret[3] += self.rwd[3]                                
                            
                            # Set action mask
                            self.mask[3] = [0]*self.n_loc_sbox
                            for x in l_cand_sbox: self.mask[3][x] = 1

                            # Get RL states 
                            obs_list = [(self.n_sorted_items-100)/20, self.probo2.ny*2-4, self.probo2.dir*2-4]
                            l_loaded_items_rotc = self.rotc.GetLoadedItemArray(time_mod_rotc)
                            for i in range(len(self.alloc_sboxes)):
                                if i not in l_cand_sbox:
                                    obs_list.extend([-1]*self.n_item_load)
                                else:
                                    l_unsorted_items =list(self.alloc_sboxes[i].unsorted_items.keys())
                                    for rotc_item in l_loaded_items_rotc:
                                        if rotc_item == '': obs_list.append(-1)
                                        elif rotc_item in l_unsorted_items: obs_list.append(1)
                                        else: obs_list.append(-1)
                                obs_list.append(sum(list(self.alloc_sboxes[i].unsorted_items.values())))

                            self.old_obs[3] = obs_list

                            if self.marl_type == 'cdsc' or self.marl_type == 'cdic':
                              obs_vf_list = self.get_all_states(3, -1)
                            
                            if self.marl_type == 'cdsc':
                                self.act[3] = self.rl[3].select_action(self.old_obs[3]+self.mask[3]+[self.time/5e4], self.time, self.mask[3])
                                state_value = self.rlCritic.estimate_vf(obs_vf_list)
                            elif self.marl_type == 'cdic':
                                self.act[3], state_value = self.rl[3].select_action(self.old_obs[3]+self.mask[3]+[self.time/5e4], obs_vf_list, self.time, self.mask[3])
                            else:
                                self.act[3] = self.rl[3].select_action(self.old_obs[3] + self.mask[3] + [self.time/5e4], self.time, self.mask[3])

                            self.b_do_act[3] = True
                            if self.marl_type == 'illr': self.t_act_latest[3] = self.time

                            self.all_trans_agt.append(3), self.all_frame.append(self.time)
                            if self.marl_type == 'cdsc' or self.marl_type == 'cdic': 
                                self.all_state_value.append(state_value)
                        
                        sbox_id = self.act[3]   
                
                if sbox_id != '': # Exist next shipping box to sort
                    self.probo2.status.remove('free') # Remove "free" flag if there are items to be picked
                    if self.probo2.ny != sbox_id:
                        # Set "moving" flag, translation velocity and moving time if PR2 locates different positions from target shipping box
                        self.probo2.status.append('moving')                    
                        self.probo2.t_wait_move = self.probo2.t_move * abs(sbox_id-self.probo2.ny)
                        self.probo2.vel = (self.probo2.y_pos[sbox_id] - self.probo2.y) / self.probo2.t_wait_move
                        self.probo2.ny = sbox_id  
                    else:
                        # Set "can_pick" flag
                        self.probo2.status.append('can_pick')

                    # Set sorting box to process
                    self.probo2.sort_sbox = sbox_id
                else: self.probo2.sort_sbox = ''

            # If PR2 is free and the item to be picked has already arrived, the picking operation gets started
            elif 'can_pick' in self.probo2.status:
                set_sbox_unsorted_items = set(self.alloc_sboxes[self.probo2.ny].unsorted_items)
                if len(set_sbox_unsorted_items) > 0: # Exist unsorted items
                    loaded_item = self.rotc.GetItemToPickOnRotConv(time_mod_rotc, self.probo2.ny)
                    if loaded_item != '': # Exist unsorted items on carousel conveyor
                        if loaded_item in set_sbox_unsorted_items:
                            # Remove "can_pick" flag and add "picking" flag
                            # Set picking time             
                            self.probo2.status.remove('can_pick')
                            self.probo2.status.append('picking')
                            self.probo2.t_wait_pick = self.probo2.t_pick
                            self.probo2.having_item = loaded_item
                            # Update item status
                            update_item_in_proc(loaded_item, 'rotc', 'probo2', self.in_proc_items, self.time, 
                                            self.probo2.x, self.probo2.y, '') 
                            # Update item configuration on carousel conveyor
                            self.rotc.UnloadItem(time_mod_rotc, self.probo2.ny)
                else:
                    self.probo2.status.remove('can_pick')
                    self.probo2.status.append('free')

            # Start of turning
            elif 'picking' in self.probo2.status and \
                self.probo2.t_wait_pick == 0:
                # Remove "picking" flag and add "rotating1" flag
                # Set rotation time     
                self.probo2.status.remove('picking')
                self.probo2.status.append('rotating1') 
                self.probo2.t_wait_rot = self.probo2.t_rot_pi 
                self.probo2.change_dir(0) # Change PR2 direction to upward
                
            # Start placing item          
            elif 'can_place' in self.probo2.status:
                # Remove "can_place" flag and add "placing" flag
                # Set item-placing time   
                self.probo2.status.remove('can_place')
                self.probo2.status.append('placing') 
                placetime = int(self.probo2.t_place + random.normalvariate(0.0, self.sigma)*self.fps)
                if placetime>0:self.probo2.t_wait_place = placetime
                else: self.probo2.t_wait_place = self.probo2.t_place 

            # Assing new picking task to PR2 if completed placing operation
            elif 'placing' in self.probo2.status and \
                self.probo2.t_wait_place == 0:

                placed_item_type = self.probo2.having_item
                self.probo2.having_item = ''

                # Update item status in process
                update_item_in_proc(placed_item_type, 'probo2', 'sbox', self.in_proc_items, self.time, 
                                    self.alloc_sboxes[self.probo2.ny].x, self.alloc_sboxes[self.probo2.ny].y,
                                    self.alloc_sboxes[self.probo2.ny].ship_id) 

                # Update status of shipping boxes by adding sorted-item information
                if placed_item_type not in self.alloc_sboxes[self.probo2.ny].sorted_items.keys():
                    self.alloc_sboxes[self.probo2.ny].sorted_items[placed_item_type] = \
                     [[self.alloc_sboxes[self.probo2.ny].x_items[self.alloc_sboxes[self.probo2.ny].n_sorted_items]], 
                      [self.alloc_sboxes[self.probo2.ny].y_items[self.alloc_sboxes[self.probo2.ny].n_sorted_items]]]
                else: 
                    self.alloc_sboxes[self.probo2.ny].sorted_items[placed_item_type][0].append(
                    self.alloc_sboxes[self.probo2.ny].x_items[self.alloc_sboxes[self.probo2.ny].n_sorted_items ])
                    self.alloc_sboxes[self.probo2.ny].sorted_items[placed_item_type][1].append(
                    self.alloc_sboxes[self.probo2.ny].y_items[self.alloc_sboxes[self.probo2.ny].n_sorted_items])

                self.n_sorted_items += 1 # Count up sorted item
                self.alloc_sboxes[self.probo2.ny].n_sorted_items += 1 # Count up sorted item for corresponding shipping box

                # Remove sorted-item information from shipping-box statues registered as unsorted item
                if self.alloc_sboxes[self.probo2.ny].unsorted_items[placed_item_type] == 1:
                   del self.alloc_sboxes[self.probo2.ny].unsorted_items[placed_item_type]
                   if len(self.alloc_sboxes[self.probo2.ny].unsorted_items) == 0:
                       self.alloc_sboxes[self.probo2.ny].b_completed = True
                       self.alloc_sboxes[self.probo2.ny].status = 'completed'
                       self.alloc_sboxes[self.probo2.ny].t_wait_swap = self.alloc_sboxes[self.probo2.ny].t_swap
                else: self.alloc_sboxes[self.probo2.ny].unsorted_items[placed_item_type] -= 1

                # Assign new picking task to PR2
                # Remove "placing" flag and add "rotating2" flag
                # Set rotation time   
                self.probo2.status.remove('placing')                                   
                self.probo2.status.append('rotating2') 
                self.probo2.t_wait_rot = self.probo2.t_rot_pi 
                self.probo2.change_dir(1) # Change PR2 direction to downward

                sbox_id_next, ship_id_next = '', ''
                items_rotc = self.in_proc_items['rotc']
                # Check how many boxes items can be sorted into
                l_cand_sbox = []
                set_item_type_rotc_uniq = set([item_info.type for item_info in items_rotc]) # Get unique item types on carousel conveyor
                for i in range(len(self.alloc_sboxes)):
                    if self.alloc_sboxes[i].b_completed: continue
                    set_sbox_unsorted_items = set(self.alloc_sboxes[i].unsorted_items) # Get unique item types required for i-th shipping box
                    if len(set_item_type_rotc_uniq & set_sbox_unsorted_items): l_cand_sbox.append(i)

                if self.rl[3].b_na_one and len(l_cand_sbox) == 1: # Only one available sipping box for sorting items on carousel conveyor 
                    sbox_id, ship_id = l_cand_sbox[0], self.alloc_sboxes[0].ship_id
                elif len(l_cand_sbox) > 0: # Exist more than two available shipping boxes 
                    if self.rl[3] != '': # RL case
                        if self.b_do_act[3]:                                        
                            self.all_is_terminal.append(False), self.rl[3].buffer.is_terminals.append(False)
                            self.n_buf[3] += 1
                            self.b_do_act[3] = False
                            self.rwd[3] = self.calc_immediate_reward(3)
                            self.rl[3].buffer.rewards.append(self.rwd[3])
                            self.ep_ret[3] += self.rwd[3]

                        # Set action mask
                        self.mask[3] = [0]*self.n_loc_sbox
                        for x in l_cand_sbox: self.mask[3][x] = 1

                        # Get RL states 
                        obs_list = [(self.n_sorted_items-100)/20, self.probo2.ny*2-4, self.probo2.dir*2-4]

                        l_loaded_items_rotc = self.rotc.GetLoadedItemArray(time_mod_rotc)
                        for i in range(len(self.alloc_sboxes)):
                            if i not in l_cand_sbox:
                                obs_list.extend([-1]*self.n_item_load)
                            else:
                                l_unsorted_items =list(self.alloc_sboxes[i].unsorted_items.keys())
                                for rotc_item in l_loaded_items_rotc:
                                    if rotc_item == '': obs_list.append(-1)
                                    elif rotc_item in l_unsorted_items: obs_list.append(1)
                                    else: obs_list.append(-1)

                            obs_list.append(sum(list(self.alloc_sboxes[i].unsorted_items.values())))

                        self.old_obs[3] = obs_list

                        if self.marl_type == 'cdsc' or self.marl_type == 'cdic': 
                            obs_vf_list = self.get_all_states(3, -1)
 
                        if self.marl_type == 'cdsc':
                            self.act[3] = self.rl[3].select_action(self.old_obs[3]+self.mask[3]+[self.time/5e4], self.time, self.mask[3])
                            state_value = self.rlCritic.estimate_vf(obs_vf_list)
                        elif self.marl_type == 'cdic':
                            self.act[3], state_value = self.rl[3].select_action(self.old_obs[3]+self.mask[3]+[self.time/5e4], obs_vf_list, self.time, self.mask[3])
                        else:
                            self.act[3] = self.rl[3].select_action(self.old_obs[3] + self.mask[3] + [self.time/5e4], self.time, self.mask[3])
                            
                        self.b_do_act[3] = True
                        if self.marl_type == 'illr': self.t_act_latest[3] = self.time

                        self.all_trans_agt.append(3), self.all_frame.append(self.time)
                        if self.marl_type == 'cdsc' or self.marl_type == 'cdic': 
                            self.all_state_value.append(state_value)

                    sbox_id_next = self.act[3]

                if sbox_id_next != '': 
                    self.probo2.sort_sbox = sbox_id_next # Register next shipping box to sort
                    if self.probo2.ny != sbox_id_next:
                        # Set move instruction
                        self.probo2.status.append('moving')                    
                        self.probo2.t_wait_move = self.probo2.t_move * abs(sbox_id_next-self.probo2.ny)
                        self.probo2.vel = (self.probo2.y_pos[sbox_id_next] - self.probo2.y) / self.probo2.t_wait_move
                        self.probo2.ny = sbox_id_next  
                else: self.probo2.sort_sbox = ''

        # :: ----------------------------------------------------------------
        # PR1
        # :: ----------------------------------------------------------------
        if not self.probo1.b_start: 
            # Count down to start next work
            self.probo1.t_start -= 1
            if self.probo1.t_start == 0: self.probo1.b_start = True
        else:
            if self.probo1.t_wait_move > 0 or self.probo1.t_wait_rot > 0:
                # Count down wating times 
                if self.probo1.t_wait_move > 0:
                    self.probo1.t_wait_move -= 1
                    self.probo1.y = self.probo1.y+self.probo1.vel                    
                    if self.probo1.t_wait_move == 0: # Arrive to the target conveyor 
                        self.probo1.status.remove('moving')                 # Remove state flag "moving"
                        self.probo1.y = self.probo1.y_pos[self.probo1.ny]   # Set precise position
                        self.probo1.vel = 0                                 # Set velocity to zero
                if self.probo1.t_wait_rot > 0:
                    self.probo1.t_wait_rot -= 1
                    if self.probo1.t_wait_rot == 0: # Rotation completed
                        if 'rotating1' in self.probo1.status: # Completed to turning to placing side
                            self.probo1.status.remove('rotating1')    
                            self.probo1.change_dir(3) # Change PR1 direction to placing side
                        else: # Completed to turning to picking side
                            self.probo1.status.remove('rotating2')       
                            self.probo1.change_dir(2) # Change PR2 direction to picking side

                if self.probo1.t_wait_move == 0 and self.probo1.t_wait_rot == 0:
                    # Confirm availability to pick or place items
                    if self.probo1.dir == 3: self.probo1.status.append('can_place')
                    elif not self.probo1.scheduled: self.probo1.status.append('free')
                    else: self.probo1.status.append('can_pick')
            elif self.probo1.t_wait_pick > 0:
                # Count down picking time  
                self.probo1.t_wait_pick -= 1
            elif self.probo1.t_wait_place > 0:
                # Count down placing time
                self.probo1.t_wait_place -= 1

            # Assign new picking task to PR1 if it's free
            if 'free' in self.probo1.status and not self.probo1.scheduled:
                if len(self.in_proc_items['conv']) > 0:
                    item, conv_id = '', ''
                    cur_conv_id = self.probo1.ny        # Current conveyor ID
                    d_conv_nearest_dist = {}	        # Item location closest to picking area (= end of conveyor) 
                    for i in range(len(self.convs)):
                        d_item_conv = self.convs[i].x_loaded_items
                        if len(d_item_conv) == 0: continue
                        d_conv_nearest_dist[i] = next(iter(d_item_conv))
                    if len(d_conv_nearest_dist) == 0: pass
                    elif self.rl[2].b_na_one and len(d_conv_nearest_dist) == 1: # Exist only one available conveyor for picking
                        min_conv_id =  next(iter(d_conv_nearest_dist))
                        item, conv_id = next(iter(self.convs[min_conv_id].x_loaded_items)), min_conv_id
                    elif not self.rl[2].b_na_one and len(d_conv_nearest_dist) > 0:  # Exist more than two available conveyors
                        if self.rl[2] != '': # RL case
                            if self.b_do_act[2]:                                       
                                self.all_is_terminal.append(False), self.rl[2].buffer.is_terminals.append(False)
                                self.n_buf[2] += 1
                                self.b_do_act[2] = False
                                self.rwd[2] = self.calc_immediate_reward(2)
                                self.rl[2].buffer.rewards.append(self.rwd[2])
                                self.ep_ret[2] += self.rwd[2]
                                    
                            # Set action mask
                            self.mask[2] = [0]*self.n_conv
                            for x in d_conv_nearest_dist.keys(): self.mask[2][x] = 1

                            # Get RL states
                            obs_list = [self.probo1.ny, self.probo1.dir]
                            if 'rotc' in self.in_proc_items: obs_list.extend([len(self.in_proc_items['rotc'])])
                            else:obs_list.extend([0])

                            for i in range(len(self.convs)):
                                d_item_conv = self.convs[i].x_loaded_items
                                d_item_conv = dict(sorted(self.convs[i].x_loaded_items.items(), key=itemgetter(0), reverse=True))	
                                if len(d_item_conv) == 0: # No items on conveyor
                                    obs_list.extend([-20, 0, -20, 0])
                                    continue
                                pos_item, head_item_type = next(iter(d_item_conv)), d_item_conv[next(iter(d_item_conv))] 
                                pos_item_last = min(d_item_conv)
                                n_head_items = 0
                                for k, v in d_item_conv.items():
                                    if v == head_item_type: n_head_items += 1
                                    else: break	
                                n_tot_items = len(d_item_conv.items()) - n_head_items 
                                obs_list.extend([(pos_item+2.17e4)/2.17e4*20-10, n_head_items, (pos_item_last+2.17e4)/2.17e4*20-10, n_tot_items])

                            loading_items = [] 
                            for i in range(len(self.convs)):
                                if self.convs[i].loading_item != '':
                                    l_loading_items = self.in_proc_items['loading_conv']
                                    count_loading_items = 0
                                    for itm in l_loading_items:
                                        if itm.type == self.convs[i].loading_item: count_loading_items += 1
                                    obs_list.extend([(self.convs[i].x_lock_for_loading+2.17e4)/2.17e4*20-10, count_loading_items])
                                    loading_items += [self.convs[i].loading_item]
                                else: obs_list.extend([-10, 0])

                            l_items = self.in_proc_items['backlogged']                 
                            if len((l_items)) > 0:
                                l_candi_items = [] 
                                d_items = {}
                                for j in range(len(l_items)):
                                    if l_items[j].type in loading_items: continue
                                    l_candi_items += [l_items[j]]
                                d_candi_pcs, d_candi_type = {}, {}
                                l_uniq_candi_type = []
                                for cand_item in l_candi_items:
                                    loc_candi = self.item_locations[cand_item.type]
                                    if loc_candi not in d_candi_pcs:
                                        d_candi_pcs[loc_candi], d_candi_type[loc_candi] = 1, 1
                                        l_uniq_candi_type.append(cand_item.type)
                                    else:
                                        d_candi_pcs[loc_candi] += 1
                                        if cand_item.type not in l_uniq_candi_type: 
                                            l_uniq_candi_type.append(cand_item.type)
                                            d_candi_type[loc_candi] = 1

                                for j in range(self.n_loc):
                                    if self.l_item_locations[j] not in d_candi_pcs: obs_list.extend([0.0])
                                    else: obs_list.extend([d_candi_pcs[self.l_item_locations[j]]])
                            else: obs_list.extend([0]*self.n_loc)
                            
                            self.old_obs[2] = obs_list

                            if self.marl_type == 'cdsc' or self.marl_type == 'cdic':
                                obs_vf_list = self.get_all_states(2, -1)
                            
                            if self.marl_type == 'cdsc':
                                self.act[2] = self.rl[2].select_action(self.old_obs[2]+self.mask[2]+[self.time/5e4], self.time, self.mask[2])
                                state_value = self.rlCritic.estimate_vf(obs_vf_list)
                            elif self.marl_type == 'cdic':
                                self.act[2], state_value = self.rl[2].select_action(self.old_obs[2]+self.mask[2]+[self.time/5e4], obs_vf_list, self.time, self.mask[2])
                            else:
                                self.act[2] = self.rl[2].select_action(self.old_obs[2] + self.mask[2] + [self.time/5e4], self.time, self.mask[2])

                            self.b_do_act[2] = True
                            if self.marl_type == 'illr': self.t_act_latest[2] = self.time

                            self.all_trans_agt.append(2), self.all_frame.append(self.time)
                            if self.marl_type == 'cdsc' or self.marl_type == 'cdic': 
                                self.all_state_value.append(state_value)

                        conv_id = self.act[2]
                        d_item_conv = dict(sorted(self.convs[conv_id].x_loaded_items.items(), key=itemgetter(0), reverse=True))
                        item = d_item_conv[next(iter(d_item_conv))]

                    if item != '': 
                        self.probo1.status.remove('free') # Remove "free" flag if there are items to be picked
                        self.probo1.scheduled = True
                        if self.probo1.ny != conv_id:
                            # Set "moving" flag and moving time if PR2 locates different positions from target shipping box
                            self.probo1.status.append('moving')                    
                            self.probo1.t_wait_move = self.probo1.t_move * abs(conv_id-self.probo1.ny)
                            self.probo1.vel = (self.probo1.y_pos[conv_id] - self.probo1.y) / self.probo1.t_wait_move
                            self.probo1.ny = conv_id  
                        else:
                            # Set "can_pick" flag
                            self.probo1.status.append('can_pick')

            # If PR1 is free and has already arrived at the target conveyor, the picking operation gets started
            elif 'can_pick' in self.probo1.status:                
                if self.convs[self.probo1.ny].xend in self.convs[self.probo1.ny].x_loaded_items.keys():
                    # Remove "can_pick" flag and add "picking" flag
                    # Set item-picking time              
                    self.probo1.status.remove('can_pick')
                    self.probo1.status.append('picking')
                    self.probo1.t_wait_pick = self.probo1.t_pick
                
                    # Set pick-up item
                    # PR1 continues to pick up same-type items as the top one
                    conv_id = self.probo1.ny
                    d_item_conv = dict(sorted(self.convs[conv_id].x_loaded_items.items(), key=itemgetter(0), reverse=True))
                    pick_item_id = d_item_conv[self.convs[conv_id].xend]
                    # Register head item-id on conveyor for picking up                 
                    self.probo1.pick_items[self.convs[conv_id].xend] = pick_item_id
                    # Register number of items including subsequent ones with same type 
                    for i in range(1, self.probo1.n_pick):                    
                        x_next = self.convs[conv_id].xend - i*int(self.convs[conv_id].item_intvl_conv)
                        if x_next in d_item_conv and d_item_conv[x_next] == pick_item_id: # Check location of subsequent item and item type
                            self.probo1.pick_items[x_next] = pick_item_id
                        else: break # Terminate if there is no subsequent item  
                elif not self.probo1.scheduled:
                    self.probo1.status.remove('can_pick')
                    self.probo1.status.append('free')

            # If PR1 is picking and pick operation time has elapsed, PR1 starts to rotate.
            elif 'picking' in self.probo1.status and \
                self.probo1.t_wait_pick == 0:
                # Remove "picking" flag and add "rotating1" flag
                # Set rotating time
                self.probo1.status.remove('picking')
                self.probo1.status.append('rotating1') 
                self.probo1.t_wait_rot = self.probo1.t_rot_pi 
                self.probo1.change_dir(0) # Change PR1 direction to upward
                
                # Update both the configuration and status of items on PC
                conv_id = self.probo1.ny
                d_item_conv = self.convs[conv_id].x_loaded_items
                l_item_conv = self.in_proc_items['conv']
                for k, v in self.probo1.pick_items.items():                   
                    del d_item_conv[k] # Remove information of picked item
                    update_item_in_proc(v, 'conv', 'probo1', self.in_proc_items, 
                                        self.time, self.probo1.x, self.probo1.y, "") # Update item status

                self.convs[conv_id].x_loaded_items = d_item_conv

            # If PR1 can place item and placing space on carousel conveyor is empty,
            # PR1 starts to place an item         
            elif 'can_place' in self.probo1.status and \
                not self.rotc.ExistItemOnRotConv(time_mod_rotc, self.probo1.nseq_pos):
                # Remove "can_place" flag and add "placing" flag 
                # Set item-placing time 
                self.probo1.status.remove('can_place')
                self.probo1.status.append('placing') 
                placetime = int(self.probo1.t_place + random.normalvariate(0.0, self.sigma)*self.fps)
                if placetime>0: self.probo1.t_wait_place = int(placetime)
                else: self.probo1.t_wait_place = self.probo1.t_place 

                # Register information of placing item to carousel conveyor
                self.rotc.LoadItem(time_mod_rotc, self.probo1.nseq_pos, self.probo1.pick_items[next(iter(self.probo1.pick_items))])

            # If PR1 is placing and place operation time has elapsed, 
            # PR1 repeat pick-and-place operation of same-type item or assign new picking task of different-type item.
            elif 'placing' in self.probo1.status and \
                self.probo1.t_wait_place == 0:
                # Remove information of placed item
                key_first_item = next(iter(self.probo1.pick_items))
                placed_item_type = self.probo1.pick_items[key_first_item]
                del self.probo1.pick_items[key_first_item]
                update_item_in_proc(placed_item_type, 'probo1', 'rotc', self.in_proc_items, 
                                    self.time, self.rotc.pconv_place_loc[self.probo1.ny][0], 
                                    self.rotc.pconv_place_loc[self.probo1.ny][1], '') # Update item status           

                if len(self.probo1.pick_items) > 0:
                    # Repeat placing operation if PR1 still has picked items.
                    self.probo1.status.remove('placing')
                    self.probo1.status.append('can_place') 
                else:
                    # Assign new picking task if no picked item remained.
                    # Remove "placing" flag and add "rotating2" flag
                    # Set rotating time
                    self.probo1.status.remove('placing')                                   
                    self.probo1.status.append('rotating2') 
                    self.probo1.t_wait_rot = self.probo1.t_rot_pi 
                    self.probo1.change_dir(1) # Change PR1 direction to downward

                    self.probo1.scheduled = False

                    # Select conveyor to pick up new item
                    conv_id = self.probo1.ny
                    d_item_conv = dict(sorted(self.convs[conv_id].x_loaded_items.items(), key=itemgetter(0), reverse=True))
                    if len(d_item_conv)==0 or \
                        d_item_conv[next(iter(d_item_conv))] == placed_item_type: # No item exists or different-type item is located subsequently on the current conveyor, 
                        item, conv_id_next = '', ''
                        cur_conv_id = self.probo1.ny # Current conveyor ID
                        d_conv_nearest_dist = {}	 # Item location of each conveyor nearest to PR1
                        for i in range(len(self.convs)):
                            d_item_conv = self.convs[i].x_loaded_items
                            if len(d_item_conv) == 0: continue
                            d_conv_nearest_dist[i] = next(iter(d_item_conv))
                        if len(d_conv_nearest_dist) == 0: pass
                        elif self.rl[2].b_na_one and len(d_conv_nearest_dist) == 1:
                            min_conv_id =  next(iter(d_conv_nearest_dist))
                            item, conv_id_next = next(iter(self.convs[min_conv_id].x_loaded_items)), min_conv_id
                        elif not self.rl[2].b_na_one and len(d_conv_nearest_dist)>0: 
                            #  Exist more than two available conveyors
                            if self.rl[2] != '':
                                if self.b_do_act[2]:
                                    self.all_is_terminal.append(False), self.rl[2].buffer.is_terminals.append(False)
                                    self.n_buf[2] += 1
                                    self.b_do_act[2] = False
                                    self.rwd[2] = self.calc_immediate_reward(2)    
                                    self.rl[2].buffer.rewards.append(self.rwd[2])
                                    self.ep_ret[2] += self.rwd[2]

                                # Action mask
                                self.mask[2] = [0]*self.n_conv
                                for x in d_conv_nearest_dist.keys(): self.mask[2][x] = 1

                                # Get RL states
                                obs_list = [self.probo1.ny, self.probo1.dir]
                                if 'rotc' in self.in_proc_items: obs_list.extend([len(self.in_proc_items['rotc'])])
                                else:obs_list.extend([0])

                                for i in range(len(self.convs)):
                                    d_item_conv = self.convs[i].x_loaded_items
                                    d_item_conv = dict(sorted(self.convs[i].x_loaded_items.items(), key=itemgetter(0), reverse=True))	
                                    if len(d_item_conv) == 0:
                                        obs_list.extend([-20, 0, -20, 0])
                                        continue
                                    pos_item, head_item_type = next(iter(d_item_conv)), d_item_conv[next(iter(d_item_conv))]
                                    pos_item_last = min(d_item_conv)

                                    n_head_items = 0
                                    for k, v in d_item_conv.items():
                                        if v == head_item_type: n_head_items += 1
                                        else: break	
                                    n_tot_items = len(d_item_conv.items()) - n_head_items 
                                    obs_list.extend([(pos_item+2.17e4)/2.17e4*20-10, n_head_items, (pos_item_last+2.17e4)/2.17e4*20-10, n_tot_items])

                                loading_items = [] 
                                for i in range(len(self.convs)):
                                    if self.convs[i].loading_item != '':
                                        l_loading_items = self.in_proc_items['loading_conv']
                                        count_loading_items = 0
                                        for itm in l_loading_items:
                                            if itm.type == self.convs[i].loading_item: count_loading_items += 1
                                        obs_list.extend([(self.convs[i].x_lock_for_loading+2.17e4)/2.17e4*20-10, count_loading_items])
                                        loading_items += [self.convs[i].loading_item]
                                    else:
                                        obs_list.extend([-10, 0])

                                l_items = self.in_proc_items['backlogged']                 
                                if len((l_items)) > 0:
                                    l_candi_items = [] 
                                    d_items = {}
                                    for j in range(len(l_items)):
                                        if l_items[j].type in loading_items: continue
                                        l_candi_items += [l_items[j]]

                                    d_candi_pcs, d_candi_type = {}, {}
                                    l_uniq_candi_type = []
                                    for cand_item in l_candi_items:
                                        loc_candi = self.item_locations[cand_item.type]
                                        if loc_candi not in d_candi_pcs:
                                            d_candi_pcs[loc_candi] = 1
                                            d_candi_type[loc_candi] = 1
                                            l_uniq_candi_type.append(cand_item.type)
                                        else:
                                            d_candi_pcs[loc_candi] += 1
                                            if cand_item.type not in l_uniq_candi_type: 
                                                l_uniq_candi_type.append(cand_item.type)
                                                d_candi_type[loc_candi] = 1

                                    for j in range(self.n_loc):
                                        if self.l_item_locations[j] not in d_candi_pcs:
                                            obs_list.extend([0.0])
                                        else:
                                            obs_list.extend([d_candi_pcs[self.l_item_locations[j]]])
                                else:
                                    obs_list.extend([0] * self.n_loc)

                                self.old_obs[2] = obs_list

                                if self.marl_type == 'cdsc' or self.marl_type == 'cdic':
                                    obs_vf_list = self.get_all_states(2, -1)
                                
                                if self.marl_type == 'cdsc':
                                    self.act[2] = self.rl[2].select_action(self.old_obs[2]+self.mask[2]+[self.time/5e4], self.time, self.mask[2])
                                    state_value = self.rlCritic.estimate_vf(obs_vf_list)
                                elif self.marl_type == 'cdic':
                                    self.act[2], state_value = self.rl[2].select_action(self.old_obs[2]+self.mask[2]+[self.time/5e4], obs_vf_list, self.time, self.mask[2])
                                else:
                                    self.act[2] = self.rl[2].select_action(self.old_obs[2] + self.mask[2] + [self.time/5e4], self.time, self.mask[2])
    
                                self.b_do_act[2] = True
                                if self.marl_type == 'illr': self.t_act_latest[2] = self.time
                                
                                self.all_trans_agt.append(2), self.all_frame.append(self.time)
                                if self.marl_type == 'cdsc' or self.marl_type == 'cdic': 
                                    self.all_state_value.append(state_value)

                            conv_id_next = self.act[2]
                            d_item_conv = dict(sorted(self.convs[conv_id_next].x_loaded_items.items(), key=itemgetter(0), reverse=True))
                            item = d_item_conv[next(iter(d_item_conv))]

                        if item != '': # Exist next item to pick
                            self.probo1.scheduled = True
                            if self.probo1.ny != conv_id_next:
                                # If PR1 is not located at the scheduled conveyor,
                                # Set "moving" flag, translation velocity and moving time. 
                                self.probo1.status.append('moving')                            
                                self.probo1.t_wait_move = self.probo1.t_move * abs(conv_id_next-self.probo1.ny)
                                self.probo1.vel = (self.probo1.y_pos[conv_id] - self.probo1.y) / self.probo1.t_wait_move
                                self.probo1.ny = conv_id_next  
                    else:
                        # Repeat picking same-type item from same conveyor
                        pass

        # :: ----------------------------------------------------------------
        # :: PC
        # :: ----------------------------------------------------------------
        if 'conv' in self.in_proc_items.keys(): # Exist more than one item on PC
            for i in range(self.n_conv):
                d_sorted_x = dict(sorted(self.convs[i].x_loaded_items.items(), key=itemgetter(0), reverse=True))
                d_copy_sorted_x = deepcopy(d_sorted_x)
                d_sorted_x_new = {}                    
                for key, val in d_sorted_x.items():
                    if key == self.convs[i].xend: 
                        d_sorted_x_new[key] = val # Unchange the location of head item at the tail end of conveyor
                        continue
                    x_next = key + self.convs[i].speed_ppf
                    if x_next > self.convs[i].xend or \
                        x_next in d_sorted_x_new.keys() or \
                        self.convs[i].ExistLockedX(key, x_next):  
                        # If item location at the next time-step 
                        # (exceeds the conveyor tail end) or (exists other item) or (exits loaded item from loading port),
                        # it is advaneced as much as possible ( < self.convs[i].speed_ppf).
                        exist_neighbor = False
                        for j in range(1, int(self.convs[i].speed_ppf/self.convs[i].item_intvl_conv)):
                            x_next_nearest = x_next - j*self.convs[i].item_intvl_conv
                            if x_next_nearest > self.convs[i].xend: continue
                            if self.convs[i].ExistLockedX(key, x_next_nearest): continue
                            if x_next_nearest not in d_sorted_x_new.keys():
                                d_sorted_x_new[x_next_nearest] = val
                                exist_neighbor = True
                                del d_copy_sorted_x[key]
                                break
                        if not exist_neighbor:
                            # No empty location for next time-step. The item location is remained.
                            d_sorted_x_new[key] = val
                    else: 
                        # Update the item location calculated by simply adding one-step translation
                        d_sorted_x_new[x_next] = val
                        del d_copy_sorted_x[key]

                # Update the item location on conveyor
                self.convs[i].x_loaded_items = d_sorted_x_new

        # If location is empty on conveyor, new items are loaded one by one.
        loading_items = [] # items in loading to conveyor
        for i in range(self.n_conv):
            if self.convs[i].count_pcs_interval > 0: self.convs[i].count_pcs_interval -= 1
            if self.convs[i].intvl_loading > 0: self.convs[i].intvl_loading -= 1
            
            if self.convs[i].loading_item == '': continue # Skip if no item is expected to be installed.
            loading_items += [self.convs[i].loading_item]
            if self.convs[i].count_pcs_interval > 0: continue 
            x_loading = self.convs[i].x_entries[self.convs[i].loading_item] 
            if self.convs[i].ExistSubsequentItem(x_loading): continue       
            self.convs[i].x_loaded_items[x_loading] = self.convs[i].loading_item        
            if not self.convs[i].IsLoadingXLocked(): self.convs[i].x_lock_for_loading = x_loading            

            update_item_in_proc(self.convs[i].loading_item, 'loading_conv', 'conv', self.in_proc_items, 
                                self.time, self.convs[i].x_lock_for_loading, self.convs[i].y, '')
            putin_tvar = int(random.normalvariate(0, self.sigma)*self.fps)
            self.convs[i].SetOneItemInterval(tvar=putin_tvar) 

            hit_flg = False
            for j in range(len(self.in_proc_items['loading_conv'])):
                if self.in_proc_items['loading_conv'][j].type == self.convs[i].loading_item: 
                    hit_flg = True
                    break
            if not hit_flg: 
                self.convs[i].loading_item = ''
                self.convs[i].UnlockLoadingX()  
                intvl_tvar = int(random.normalvariate(0, self.sigma)*self.fps)
                self.convs[i].SetInterval(tvar=intvl_tvar) 

        # Select item type to load for each conveyor
        for i in range(self.n_conv):            
            if self.convs[i].loading_item == '' and \
                self.convs[i].intvl_loading == 0 and \
                self.convs[i].count_pcs_interval == 0: 
                # (there is no item loading operation) and
                # (intervals required for loading next same-type item or next item type are elapsed)  

                l_items = self.in_proc_items['backlogged']                 
                if len((l_items)) > 0:
                    l_candi_items = [] # Extract candidate items to load into conveyors
                    d_items = {}
                    for j in range(len(l_items)):
                        if l_items[j].type in loading_items: continue 
                        l_candi_items += [l_items[j]]

                        if l_items[j].type not in d_items.keys(): d_items[l_items[j].type] = 1
                        else: d_items[l_items[j].type] += 1

                    if len(l_candi_items) > 0:
                        if self.rl[1].b_na_one and len(l_candi_items) == 1: 
                            self.convs[i].loading_item = l_candi_items[0].type
                        else:
                            # Exist more than two item types for loading into conveyor 
                            if self.rl[1] != '': 
                                self.mask[1] = [0] * self.n_type
                                for x in range(len(l_items)):
                                    if l_items[x].type in loading_items: continue
                                    self.mask[1][self.uniq_type.index(l_items[x].type)] = 1

                                if self.rl[1].b_na_one and sum(self.mask[1])==1: 
                                    self.act[1] = self.mask[1].index(1)
                                else:
                                    if self.b_do_act[1]:
                                        self.all_is_terminal.append(False), self.rl[1].buffer.is_terminals.append(False)
                                        self.n_buf[1] += 1
                                        self.b_do_act[1] = False
                                        self.rwd[1] = self.calc_immediate_reward(1)
                                        self.ep_ret[1] += self.rwd[1]
                                        self.rl[1].buffer.rewards.append(self.rwd[1])

                                    # Get RL states
                                    obs_list = [i]
                                    for j in range(self.n_conv):
                                        d_item_conv = self.convs[j].x_loaded_items
                                        if len(d_item_conv) == 0: 
                                            obs_list.extend([0, 0, 0])
                                        else:
                                            l_loaded_items = list(d_item_conv.keys())
                                            obs_list.extend([(min(l_loaded_items)+2.17e4)/2.17e4*20-10, (max(l_loaded_items)+2.17e4)/2.17e4*20-10, len(d_item_conv)])

                                    d_candi_pcs, d_candi_type = {}, {}
                                    l_uniq_candi_type = []
                                    for cand_item in l_candi_items:
                                        loc_candi = self.item_locations[cand_item.type]
                                        if loc_candi not in d_candi_pcs:
                                            d_candi_pcs[loc_candi] = 1
                                            d_candi_type[loc_candi] = 1
                                            l_uniq_candi_type.append(cand_item.type)
                                        else:
                                            d_candi_pcs[loc_candi] += 1
                                            if cand_item.type not in l_uniq_candi_type: 
                                                l_uniq_candi_type.append(cand_item.type)
                                                d_candi_type[loc_candi] = 1

                                    for j in range(self.n_loc):
                                        if self.l_item_locations[j] not in d_candi_pcs:
                                            obs_list.extend([0.0, 0.0])
                                        else:
                                            obs_list.extend([d_candi_pcs[self.l_item_locations[j]], d_candi_type[self.l_item_locations[j]]])

                                    for j in range(self.n_conv):
                                        if j == i or self.convs[j].loading_item == '':
                                            obs_list.extend([0.0, 0.0])
                                        else:
                                            l_loading_items = self.in_proc_items['loading_conv']
                                            count_loading_items = 0
                                            for itm in l_loading_items:
                                                if itm.type == self.convs[j].loading_item: count_loading_items+=1
                                            obs_list.extend([self.l_item_locations.index(self.item_locations[l_loading_items[0].type]), count_loading_items])

                                    self.old_obs[1] = obs_list

                                    if self.marl_type == 'cdsc' or self.marl_type == 'cdic': 
                                        obs_vf_list = self.get_all_states(1, i)
 
                                    if self.marl_type == 'cdsc':
                                        self.act[1] = self.rl[1].select_action(self.old_obs[1]+self.mask[1]+[self.time/5e4], self.time, self.mask[1])
                                        state_value = self.rlCritic.estimate_vf(obs_vf_list)
                                    elif self.marl_type == 'cdic':
                                        self.act[1], state_value = self.rl[1].select_action(self.old_obs[1]+self.mask[1]+[self.time/5e4], obs_vf_list, self.time, self.mask[1])
                                    else:
                                        self.act[1] = self.rl[1].select_action(self.old_obs[1] + self.mask[1] + [self.time/5e4], self.time, self.mask[1])                                   

                                    self.b_do_act[1] = True
                                    if self.marl_type == 'illr': self.t_act_latest[1] = self.time

                                    self.all_trans_agt.append(1), self.all_frame.append(self.time)
                                    if self.marl_type == 'cdsc' or self.marl_type == 'cdic':
                                        self.all_state_value.append(state_value)   

                            self.convs[i].loading_item = self.uniq_type[self.act[1]]

                    if self.convs[i].loading_item != '':
                        # Update information on loading item
                        l_items_redued, l_items_selected = [], []
                        for j in range(len(l_items)):
                            if l_items[j].type==self.convs[i].loading_item:
                                l_items[j].t_in += [self.time]
                                l_items_selected += [l_items[j]]
                            else: l_items_redued += [l_items[j]]

                        self.in_proc_items['backlogged'] = l_items_redued
                        for j in range(len(l_items_selected)): l_items_selected[j].status = 'loading_conv'
                        if 'loading_conv' not in self.in_proc_items.keys(): self.in_proc_items['loading_conv'] = l_items_selected
                        else: self.in_proc_items['loading_conv'] += l_items_selected

        # Advance one time step
        self.time += 1 

        return False



