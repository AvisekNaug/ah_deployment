"""
This script containts methods that will observe the current state of the Alumni Hall
and issue a temperature set point to be sent as the set point for the building.
"""

import numpy as np
from pandas import read_csv, to_datetime
import json
from datetime import datetime, timedelta
from multiprocessing import Event, Lock
import time

import warnings
with warnings.catch_warnings():
	from stable_baselines import PPO2
from CoolProp.HumidAirProp import HAPropsSI

from alumni_scripts import data_process as dp
from alumni_scripts import alumni_data_utils as a_utils

def deploy_control(*args, **kwargs):
	"""
	Generates the actual actions to be sent to the actual building


	"""
	with open('auths.json', 'r') as fp:
		api_args = json.load(fp)
	with open('alumni_scripts/meta_data.json', 'r') as fp:
		meta_data_ = json.load(fp)

	agent_weights_available : Event = kwargs['agent_weights_available']  # deploy loop can read the agent weights now
	agent_weights_lock : Lock = kwargs['agent_weights_lock']  # prevent data read/write access
	
	# check variables if needed
	obs_space_vars : list = kwargs['obs_space_vars']
	scaler : a_utils.dataframescaler = kwargs['scaler']
	stpt_delta = np.array([0.0]) # in delta F
	stpt = np.array([68])  # in F
	stpt_scaled = scaler.minmax_scale(stpt, ['sat'], ['sat'])

	buffer_ = None  # has to be a dequeue class of length 6 maybe

	while True:
		if agent_weights_available.is_set():
			with agent_weights_lock:
				rl_agent = PPO2.load(kwargs['best_rl_agent_path'])
			agent_weights_available.clear()
			break

	while True:
	
		# get curretn observation
		df = get_real_obs(api_args, meta_data_,obs_space_vars)
		curr_obs = df.to_numpy().flatten()

		# if we want to set the sat to the exact value from previous time step
		# comment it out if not
		curr_obs[-1] = stpt_scaled.flatten()[0]

		# check individual values to lie in appropriate range


		# check individual values to not move too much from previous value


		# if they fail use average of last 4 observations from the buffer maybe


		# substitue the sat with the sat from last predict


		# add it to a queue with a buffer length of 3 hours maybe


		# predict new delta and add it to new temp var for next loop check
		stpt_delta = rl_agent.predict(curr_obs)
		stpt[0] = stpt[0] + stpt_delta[0]
		stpt_scaled = scaler.minmax_scale(stpt, ['sat'], ['sat'])

		# write it to a file



		time.sleep(timedelta(minutes=30).seconds)

	raise NotImplementedError



def get_real_obs(api_args: dict, meta_data_: dict, obs_space_vars : list):

	# arguements for the api query
	time_args = {'trend_id' : '2681', 'save_path' : 'data/trend_data/alumni_data_deployment.csv'}
	start_fields = ['start_'+i for i in ['year','month','day', 'hour', 'minute', 'second']]
	end_fields = ['end_'+i for i in ['year','month','day', 'hour', 'minute', 'second']]
	end_time = datetime.now()
	start_time = end_time - timedelta(minutes=30)
	for idx, i in enumerate(start_fields):
		time_args[i] = start_time.timetuple()[idx]
	for idx, i in enumerate(end_fields):
		time_args[i] = end_time.timetuple()[idx]
	api_args.update(time_args)

	# pull the data into csv file
	dp.pull_online_data(**api_args)

	# get the dataframe from a csv
	df_ = read_csv('data/trend_data/alumni_data_deployment.csv', )
	df_['time'] = to_datetime(df_['time'])
	df_.set_index(keys='time',inplace=True, drop = True)
	df_ = a_utils.dropNaNrows(df_)

	# add wet bulb temperature to the data
	rh = df_['WeatherDataProfile humidity']/100
	rh = rh.to_numpy()
	t_db = 5*(df_['AHU_1 outdoorAirTemp']-32)/9 + 273.15
	t_db = t_db.to_numpy()
	T = HAPropsSI('T_wb','R',rh,'T',t_db,'P',101325)
	t_f = 9*(T-273.15)/5 + 32
	df_['wbt'] = t_f

	# clean the data
	df_cleaned = dp.online_data_clean(
		meta_data_ = meta_data_, df = df_
	)
	# if data is removed substitute with some normal values from the buffer_

	# rename the columns
	new_names = []
	for i in df_cleaned.columns:
		new_names.append(meta_data_["reverse_col_alias"][i])
	df_cleaned.columns = new_names

	# clip less than 0 values
	df_cleaned.clip(lower=0)

	# aggregate data
	rolling_sum_target, rolling_mean_target = [], []
	for col_name in df_cleaned.columns:
		if meta_data_['column_agg_type'][col_name] == 'sum' : rolling_sum_target.append(col_name)
		else: rolling_mean_target.append(col_name)
	df_cleaned[rolling_sum_target] =  a_utils.window_sum(df_cleaned, window_size=6, column_names=rolling_sum_target)
	df_cleaned[rolling_mean_target] =  a_utils.window_mean(df_cleaned, window_size=6, column_names=rolling_mean_target)
	df_cleaned = a_utils.dropNaNrows(df_cleaned)

	# Sample the last half hour data
	df_cleaned = df_cleaned.iloc[[-1],:]

	# scale the columns: here we will use min-max
	scaler = a_utils.dataframescaler(meta_data_['column_stats_half_hour'])
	df_cleaned[df_cleaned.columns] = scaler.minmax_scale(df_cleaned, df_cleaned.columns, df_cleaned.columns)

	# create avg_stpt column
	stpt_cols = [ele for ele in df_cleaned.columns if 'vrf' in ele]
	df_cleaned['avg_stpt'] = df_cleaned[stpt_cols].mean(axis=1)
	# drop individual set point cols
	df_cleaned.drop( columns = stpt_cols, inplace = True)

	# rearrange observation cols
	df_cleaned = df_cleaned[obs_space_vars]

	return df_cleaned




