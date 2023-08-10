import csv
import numpy as np
import math
from copy import deepcopy
import pandas as pd
from typing import List, Dict
from sim_env.specs import *


def read_data(fpath):

	return pd.read_csv(fpath, sep=',')


def regist_item_in_proc(n_item, status, count_item, itm_type, d_item, in_time, itm_x=0, itm_y=0):

	l_items = [Item(itm_id=count_item+i, itm_t_in=in_time, itm_status=status, itm_type=itm_type, itm_x=itm_x, itm_y=itm_y) 
			for i in range(n_item)]

	if status in d_item.keys(): d_item[status] = d_item[status] + l_items
	else: d_item[status] = l_items

	return len(l_items) 
	

def update_item_in_proc(item_type, status_old, status_new, d_status_item:Dict, 
						in_time, x_new, y_new, sbox_id):

	l_item_old = d_status_item[status_old]
	l_item_new = []
	if status_new in d_status_item.keys(): 
		l_item_new = d_status_item[status_new]
		
	if len(l_item_old) == 0: print(in_time, item_type, status_old, status_new)

	for i in range(len(l_item_old)):
		if l_item_old[i].type == item_type:
			
			l_item_old[i].t_in += [in_time]
			l_item_old[i].x, l_item_old[i].y = x_new, y_new
			if sbox_id != '': l_item_old[i].sorted_box = sbox_id
			l_item_new += [l_item_old[i]]
			del l_item_old[i]
			break

	d_status_item[status_old] = l_item_old
	d_status_item[status_new] = l_item_new
