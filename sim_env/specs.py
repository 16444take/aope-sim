import numpy as np
import math
from copy import deepcopy

# Image coorinate of picking robot directing to left side
ximg_probo_l = 16
yimg_probo_l = 64

ximg_probo_r = 32
yimg_probo_r = 80

ximg_probo_u = 0
yimg_probo_u = 80

ximg_probo_d = 16
yimg_probo_d = 80

ximg_parm_l = 39
yimg_parm_l = 70
xpix_parm_l = 9
ypix_parm_l = 4

ximg_parm_r = 48
yimg_parm_r = 70
xpix_parm_r = 9
ypix_parm_r = 4

ximg_parm_u = 54
yimg_parm_u = 87
xpix_parm_u = 4
ypix_parm_u = 9

ximg_parm_d = 54
yimg_parm_d = 96
xpix_parm_d = 4
ypix_parm_d = 9

xpos_parm_l = -9
ypos_parm_l = 6

xpos_parm_r = 16
ypos_parm_r = 6

xpos_parm_u = 6
ypos_parm_u = -9

xpos_parm_d = 6
ypos_parm_d = 16


class Shipment:
    def __init__(self, ship_id, order_dic, ship_status):
        self.id = ship_id
        self.orders = orders_dic 
        self.status = ship_status

class Order:
    def __init__(self, order_id, ship_id, item_id, item_pcs, order_status):
        self.order_id = order_id
        self.ship_id = ship_id
        self.item_id = item_id 
        self.item_pcs = item_pcs
        self.status = order_status

class Item:
    def __init__(self, itm_id, itm_type, itm_t_in, itm_status, itm_x=0, itm_y=0, sorted_box=''):

        self.id = itm_id
        self.type = itm_type
        self.t_in = [itm_t_in] # time when item is loaded into each process
        self.status = itm_status
        self.x = itm_x 
        self.y = itm_y
        self.sorted_box = sorted_box

class Sbox:
    def __init__(self, box_id, ship_id, box_t_in, box_status, box_x, box_y, unsorted_items, t_swap):

        self.id = box_id
        self.ship_id = ship_id 
        self.t_in = box_t_in 
        self.status = box_status
        self.x = box_x
        self.y = box_y
        self.sorted_items = {} # key: item type,  val: [[x], [y]]
        self.unsorted_items = unsorted_items # key: item type,  val: number of items
        self.b_completed = False 
        self.t_swap = t_swap
        self.n_sorted_items = 0
        self.t_wait_swap = 0
        self.speed = (500-self.x) / self.t_swap
        self.x_items, self.y_items = [], []
        for i in range(100):
            self.x_items.append(1+6*(int(i/2)%6))
            self.y_items.append(1+6*(i%2))

    def EmptySbox(self): 
        self.id = ''
        self.ship_id = ''
        self.t_in = ''
        self.sorted_items = {} 

    def GetNumUnsortedItems(self):
        num = 0
        for k, v in self.unsorted_items.items(): num += v
        return num


class PickRobo:
    def __init__(self, robo_id, n_pick, t_pick, t_place, t_rot_pi, t_move, x_pos, y_pos, t_start,
                 t_wait_rot=0, t_wait_move=0, t_wait_pick=0, t_wait_place=0, status='free', nseq_pos=0, b_start=False):

        self.id = robo_id           
        self.dir = 2 
        self.imgx_body = ximg_probo_l 
        self.imgy_body = yimg_probo_l
        self.imgx_arm = ximg_parm_l
        self.imgy_arm = yimg_parm_l
        self.pixx_arm = xpix_parm_l
        self.pixy_arm = ypix_parm_l
        self.posx_arm = xpos_parm_l
        self.posy_arm = ypos_parm_l 

        self.n_pick = n_pick                
        self.t_pick = t_pick                
        self.t_place = t_place              
        self.t_wait_rot = t_wait_rot        
        self.t_wait_move = t_wait_move      
        self.t_wait_pick = t_wait_pick      
        self.t_wait_place = t_wait_place    
        self.t_rot_pi = t_rot_pi            
        self.t_move = t_move                
        self.status = [status]              
        self.nseq_pos = nseq_pos            
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.nx = 0
        self.ny = 0
        self.x = x_pos[self.nx]
        self.y = y_pos[self.ny]
        self.b_start = b_start
        self.t_start = t_start
        self.pick_items = {}                
        self.having_item = ''
        self.sort_sbox = 0                  
        self.pick_conv = 0                  
        self.scheduled = False

    def change_dir(self, int_dir):
        # Direction of picking robot
        # 2:left, 0:up, 3:right, 1:down

        self.dir = int_dir

        if int_dir == 2:
            self.dir = 2
            self.imgx_body = ximg_probo_l 
            self.imgy_body = yimg_probo_l
            self.imgx_arm = ximg_parm_l
            self.imgy_arm = yimg_parm_l 
            self.pixx_arm = xpix_parm_l
            self.pixy_arm = ypix_parm_l
            self.posx_arm = xpos_parm_l
            self.posy_arm = ypos_parm_l 
        elif int_dir == 3:
            self.dir = 3
            self.imgx_body = ximg_probo_r 
            self.imgy_body = yimg_probo_r
            self.imgx_arm = ximg_parm_r
            self.imgy_arm = yimg_parm_r 
            self.pixx_arm = xpix_parm_r
            self.pixy_arm = ypix_parm_r
            self.posx_arm = xpos_parm_r
            self.posy_arm = ypos_parm_r 
        elif int_dir == 0:
            self.dir = 0
            self.imgx_body = ximg_probo_u 
            self.imgy_body = yimg_probo_u
            self.imgx_arm = ximg_parm_u
            self.imgy_arm = yimg_parm_u 
            self.pixx_arm = xpix_parm_u
            self.pixy_arm = ypix_parm_u
            self.posx_arm = xpos_parm_u
            self.posy_arm = ypos_parm_u 
        else:
            self.dir = 1
            self.imgx_body = ximg_probo_d 
            self.imgy_body = yimg_probo_d
            self.imgx_arm = ximg_parm_d
            self.imgy_arm = yimg_parm_d 
            self.pixx_arm = xpix_parm_d
            self.pixy_arm = ypix_parm_d
            self.posx_arm = xpos_parm_d
            self.posy_arm = ypos_parm_d 


class Conveyor:
    def __init__(self, conv_id, conv_xend, conv_y, put_interval, item_intvl_conv, speed_ppf, 
                 num_locations_conv, loading_item='', intvl_loading=0):

        self.id = conv_id
        self.xend = conv_xend
        self.y = conv_y
        self.put_interval = put_interval
        self.item_intvl_conv = item_intvl_conv
        self.speed_ppf = speed_ppf
        self.loading_item = loading_item                
        self.num_locations_conv = num_locations_conv
        self.intvl_loading = intvl_loading

        self.locx_conv = [self.xend - self.item_intvl_conv*i for i in range(self.num_locations_conv+1)]

        self.x_loaded_items = {}
        self.x_entries = {}

        self.x_lock_for_loading = 256
        self.count_pcs_interval = 0
        self.count_interval = 0

    def UnlockLoadingX(self):
        self.x_lock_for_loading = 256

    def IsLoadingXLocked(self):
        if self.x_lock_for_loading != 256: return True
        return False

    def ExistLockedX(self, x_cur, x_next):
        if x_cur < self.x_lock_for_loading and self.x_lock_for_loading <= x_next:
            return True
        else: return False

    def SetInterval(self, tvar=0):
        if self.put_interval + tvar > 0:
            self.count_interval = self.intvl_loading + tvar
        else:
            self.count_interval = self.intvl_loading        

    def SetOneItemInterval(self, tvar=0):
        if self.put_interval + tvar > 0:
            self.count_pcs_interval = self.put_interval + tvar
        else: 
            self.count_pcs_interval = self.put_interval

    def ExistSubsequentItem(self, x_cur):
        for k in self.x_loaded_items:
            if x_cur >= k: return True
        return False


class RotConveyor:
    def __init__(self, conv_id, n_item_load, t_stay_in_same_loc, n_item=0):

        self.id = conv_id
        self.n_item_load = n_item_load 
        self.t_stay_in_same_loc = t_stay_in_same_loc       
        self.n_rotc = n_item          

        self.pos_offset_rotc = 5
        self.l_rotc = 104 + self.pos_offset_rotc
        self.r_rotc = 160 + self.pos_offset_rotc
        self.u_rotc = 24 + self.pos_offset_rotc
        self.d_rotc = 88 + self.pos_offset_rotc

        self.item_interval_x = int((self.r_rotc-self.l_rotc)/(self.n_item_load/4))
        self.item_interval_y = int((self.d_rotc-self.u_rotc)/(self.n_item_load/4))

        self.pconv_place_loc = {} 
        self.pconv_pick_loc = {} 

        self.pos_loaded_rotc = {} 
        self.time_loaded_rotc = {}

        w_rotc = self.r_rotc - self.l_rotc
        h_rotc = self.d_rotc - self.u_rotc
        
        self.pos_x_rotc, self.pos_y_rotc = {}, {} 
        for i in range(self.n_item_load): 
            for j in range(self.t_stay_in_same_loc): 
                pos_x_rotc, pos_y_rotc = [], []
                for k in range(self.n_item_load): 
                    k_move = (k+i) % self.n_item_load
                    if k_move >= 0 and k_move < int(self.n_item_load/4):
                        pos_x_rotc.append(int(self.l_rotc + k_move *self.item_interval_x))
                        pos_y_rotc.append(self.u_rotc)

                    elif k_move >= int(self.n_item_load/4) and k_move < int(2*self.n_item_load/4):
                        pos_x_rotc.append(self.r_rotc)
                        pos_y_rotc.append(int(self.u_rotc + int(k_move-int(self.n_item_load/4))*self.item_interval_y))

                    elif k_move >= int(2*self.n_item_load/4) and k_move < int(3*self.n_item_load/4):
                        pos_x_rotc.append(int(self.r_rotc - int(k_move-int(2*self.n_item_load/4))*self.item_interval_x))
                        pos_y_rotc.append(self.d_rotc)

                    else:
                        pos_x_rotc.append(self.l_rotc)
                        pos_y_rotc.append(int(self.d_rotc - int(k_move-int(3*self.n_item_load/4))*self.item_interval_y))

                self.pos_x_rotc[i*self.t_stay_in_same_loc + j] = pos_x_rotc
                self.pos_y_rotc[i*self.t_stay_in_same_loc + j] = pos_y_rotc


    def SetPlaceLocations(self, conv: Conveyor):

        conv_id = conv.id
        conv_x = conv.xend
        conv_y = conv.y

        pos_x_rotc, pos_y_rotc = self.pos_x_rotc[0], self.pos_y_rotc[0]
        dist_min, i_min = int(1e5), 0
        for i in range(len(pos_x_rotc)):
            dist = int(abs(pos_x_rotc[i] - conv_x) + abs(pos_y_rotc[i] - conv_y))
            if dist_min > dist: dist_min, i_min = dist, i

        self.pconv_place_loc[conv_id] = [pos_x_rotc[i_min], pos_y_rotc[i_min]]


    def GetLoadedItemsOnRotc(self):
        dic_loaded_items_rotc = {} 
        for k, v in self.pos_loaded_rotc.items():
            if v not in dic_loaded_items_rotc: dic_loaded_items_rotc[v] = 1
            else: dic_loaded_items_rotc[v] += 1
        return dic_loaded_items_rotc


    def GetLoadedItemArray(self, t_mod):
        pos_x_rotc_0, pos_y_rotc_0 = self.pos_x_rotc[0], self.pos_y_rotc[0]
        pos_x_rotc_t, pos_y_rotc_t = self.pos_x_rotc[t_mod], self.pos_y_rotc[t_mod]
        dic_loaded_posx, dic_loaded_posy = {}, {} 

        for k in self.pos_loaded_rotc.keys():
            dic_loaded_posx[k], dic_loaded_posy[k] = pos_x_rotc_t[k], pos_y_rotc_t[k]

        item_array = []
        for i in range(len(pos_x_rotc_0)):
            b_exist = False
            for k, v in dic_loaded_posx.items():
                if pos_x_rotc_0[i] == v and pos_y_rotc_0[i] == dic_loaded_posy[k]: 
                    item_array.append(self.pos_loaded_rotc[k])
                    b_exist = True
                    break
            if not b_exist: item_array.append('')

        return item_array


    def GetLoadedItemArray01(self, t_mod):
        pos_x_rotc_0, pos_y_rotc_0 = self.pos_x_rotc[0], self.pos_y_rotc[0]
        pos_x_rotc_t, pos_y_rotc_t = self.pos_x_rotc[t_mod], self.pos_y_rotc[t_mod]
        dic_loaded_posx, dic_loaded_posy = {}, {} 

        for k in self.pos_loaded_rotc.keys():
            dic_loaded_posx[k], dic_loaded_posy[k] = pos_x_rotc_t[k], pos_y_rotc_t[k]

        item_array = []
        for i in range(len(pos_x_rotc_0)):
            b_exist = False
            for k, v in dic_loaded_posx.items():
                if pos_x_rotc_0[i] == v and pos_y_rotc_0[i] == dic_loaded_posy[k]: 
                    item_array.append(1)
                    b_exist = True
                    break
            if not b_exist: item_array.append(0)

        return item_array


    def SetPickLocations(self, probo: PickRobo):

        for i in range(len(probo.x_pos)):
            sbox_x = probo.x_pos[i]
            sbox_y = probo.y_pos[i]

            pos_x_rotc, pos_y_rotc = self.pos_x_rotc[0], self.pos_y_rotc[0]
            dist_min, j_min = int(1e5), 0
            for j in range(len(pos_x_rotc)):
                dist = int(abs(pos_x_rotc[j] - sbox_x) + abs(pos_y_rotc[j] - sbox_y))
                if dist_min > dist: dist_min, j_min = dist, j

            self.pconv_pick_loc[i] = [pos_x_rotc[j_min], pos_y_rotc[j_min]]


    def LoadItem(self, t_mod, conv_id, item_type, time=0):
        pos_x_rotc, pos_y_rotc = self.pos_x_rotc[t_mod], self.pos_y_rotc[t_mod]
        for i in range(len(pos_x_rotc)):
            if pos_x_rotc[i] == self.pconv_place_loc[conv_id][0] and \
                pos_y_rotc[i] == self.pconv_place_loc[conv_id][1]:
                self.pos_loaded_rotc[i] = item_type
                self.time_loaded_rotc[i] = time
                break


    def UnloadItem(self, t_mod, sbox_id):
        pos_x_rotc, pos_y_rotc = self.pos_x_rotc[t_mod], self.pos_y_rotc[t_mod]
        for i in range(len(pos_x_rotc)):
            if pos_x_rotc[i] == self.pconv_pick_loc[sbox_id][0] and \
                pos_y_rotc[i] == self.pconv_pick_loc[sbox_id][1]:
                prev_time = deepcopy(self.time_loaded_rotc[i])
                del self.pos_loaded_rotc[i]
                del self.time_loaded_rotc[i]

                return prev_time


    def ExistItemOnRotConv(self, t_mod, conv_id):
        pos_x_rotc, pos_y_rotc = self.pos_x_rotc[t_mod], self.pos_y_rotc[t_mod]
        for i in range(len(pos_x_rotc)):
            if pos_x_rotc[i] == self.pconv_place_loc[conv_id][0] and \
                pos_y_rotc[i] == self.pconv_place_loc[conv_id][1]:

                if i in self.pos_loaded_rotc.keys(): return True 
                else: return False 


    def ExistItemTypeOnRotConv(self, t_mod, sbox_id, item_type):
        pos_x_rotc, pos_y_rotc = self.pos_x_rotc[t_mod], self.pos_y_rotc[t_mod]
        for i in range(len(pos_x_rotc)):
            if pos_x_rotc[i] == self.pconv_pick_loc[sbox_id][0] and \
                pos_y_rotc[i] == self.pconv_pick_loc[sbox_id][1]:

                if i in self.pos_loaded_rotc.keys() and \
                   self.pos_loaded_rotc[i] == item_type: return True 
                else: return False 


    def GetItemToPickOnRotConv(self, t_mod, sbox_id):
        pos_x_rotc, pos_y_rotc = self.pos_x_rotc[t_mod], self.pos_y_rotc[t_mod]
        for i in range(len(pos_x_rotc)):
            if pos_x_rotc[i] == self.pconv_pick_loc[sbox_id][0] and \
                pos_y_rotc[i] == self.pconv_pick_loc[sbox_id][1]:

                if i in self.pos_loaded_rotc.keys():
                    return self.pos_loaded_rotc[i] 
                else: 
                    return '' 

