"""
This script containts methods that will observe the current state of the Alumni Hall
and issue a temperature set point to be sent as the set point for the building.
"""
from os import path
import numpy as np
from pandas import read_csv, to_datetime, DataFrame
import json
from datetime import datetime, timedelta
from multiprocessing import Event, Lock
import time
import pytz
from dateutil import tz

import warnings
with warnings.catch_warnings():
	from stable_baselines import PPO2
	from alumni_scripts import data_process as dp
	from alumni_scripts import alumni_data_utils as a_utils
	from CoolProp.HumidAirProp import HAPropsSI

def deploy_control(*args, **kwargs):
	"""
	Generates the actual actions to be sent to the actual building
	"""
	# logger
	log = kwargs['logger']
	try:
		with open('auths.json', 'r') as fp:
			api_args = json.load(fp)
		with open('alumni_scripts/meta_data.json', 'r') as fp:
			meta_data_ = json.load(fp)
		if not path.exists("experience.csv"):
			with open('experience.csv', 'a+') as cfile:
				cfile.write('{},{},{},{},{},{},{},{}\n'.format('time', 'oat', 'oah', 'wbt',
				'avg_stpt', 'sat', 'rlstpt', 'hist_stpt'))
			cfile.close()

		agent_weights_available : Event = kwargs['agent_weights_available']  # deploy loop can read the agent weights now
		end_learning : Event = kwargs['end_learning']
		agent_weights_lock : Lock = kwargs['agent_weights_lock']  # prevent data read/write access
		
		# check variables if needed
		obs_space_vars : list = kwargs['obs_space_vars']
		scaler : a_utils.dataframescaler = kwargs['scaler']
		stpt_delta = np.array([0.0]) # in delta F
		stpt_unscaled = np.array([68.0])  # in F
		stpt_scaled = scaler.minmax_scale(stpt_unscaled, ['sat'], ['sat'])
		not_first_loop = True
		period = kwargs['period']

		# an initial trained model has to exist
		if agent_weights_available.is_set():
			with agent_weights_lock:
				rl_agent = PPO2.load(kwargs['best_rl_agent_path'])
			agent_weights_available.clear()
			log.info('Deploy Control Module: Controller Weights Read from offline phase')
		else:
			raise FileNotFoundError

		while not end_learning.is_set():
		
			# get current scaled and uncsaled observation
			df, df_unscaled, hist_stpt = get_real_obs(api_args, meta_data_, obs_space_vars, scaler, period, log)
			curr_obs_scaled = df.to_numpy().flatten()
			curr_obs_unscaled = df_unscaled.to_numpy().flatten()

			# if we want to set the sat to the exact value from previous time step
			# comment it out if not
			if not_first_loop:
				curr_obs_scaled[-1] = stpt_scaled[0]
				curr_obs_unscaled[-1] = stpt_unscaled[0]
			else:
				not_first_loop = True

			# check individual values to lie in appropriate range
			# already done by online_data_clean method

			# check individual values to not move too much from previous value
			# nominal values already checked within online_data_clean method

			# get new agent model in case it is available
			if agent_weights_available.is_set():
				with agent_weights_lock:
					rl_agent = PPO2.load(kwargs['best_rl_agent_path'])
				agent_weights_available.clear()
				log.info('Deploy Control Module: Controller Weights Adapted')

			# predict new delta and add it to new temp var for next loop check
			stpt_delta = rl_agent.predict(curr_obs_scaled)
			log.info('Deploy Control Module: Current SetPoint: {}'.format(curr_obs_unscaled[-1]))
			log.info('Deploy Control Module: Suggested Delta: {}'.format(stpt_delta[0][0]))
			stpt_unscaled[0] = curr_obs_unscaled[-1] + float(stpt_delta[0])  # stpt_unscaled[0] 
			# clip it in case it crosses a range
			stpt_unscaled = np.clip(stpt_unscaled, np.array([65.0]), np.array([72.0]))
			# scale it
			stpt_scaled = scaler.minmax_scale(stpt_unscaled, ['sat_stpt'], ['sat_stpt'])

			# write it to a file for BdX
			with open('reheat_preheat_setpoint.txt', 'w') as cfile:
				cfile.seek(0)
				cfile.write('{}\n'.format(str(stpt_unscaled[0])))
			cfile.close()

			# write output to file for our use
			fout = np.concatenate((curr_obs_unscaled, stpt_unscaled, hist_stpt))
			with open('experience.csv', 'a+') as cfile:
				cfile.write('{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(datetime.now(), fout[0],
				fout[1], fout[2], fout[3], fout[4], fout[5], fout[6]))
			cfile.close()

			# sleep for 30 mins before next output
			time.sleep(timedelta(minutes=30).seconds)

	except Exception as e:
		log.error('Deploy Control Module: %s', str(e))
		log.debug(e, exc_info=True)



def get_real_obs(api_args: dict, meta_data_: dict, obs_space_vars : list, scaler, period, log):

	try:
		# arguements for the api query
		time_args = {'trend_id' : '2681', 'save_path' : 'data/trend_data/alumni_data_deployment.csv'}
		start_fields = ['start_'+i for i in ['year','month','day', 'hour', 'minute', 'second']]
		end_fields = ['end_'+i for i in ['year','month','day', 'hour', 'minute', 'second']]
		end_time = datetime.now(tz=pytz.utc)
		time_gap_minutes = int((period+3)*5)
		start_time = end_time - timedelta(minutes=time_gap_minutes)
		for idx, i in enumerate(start_fields):
			time_args[i] = start_time.timetuple()[idx]
		for idx, i in enumerate(end_fields):
			time_args[i] = end_time.timetuple()[idx]
		api_args.update(time_args)

		# pull the data into csv file
		try:
			dp.pull_online_data(**api_args)
			log.info('Deploy Control Module: Deployment Data obtained from API')
		except Exception:
			log.info('Deploy Control Module: BdX API could not get data: will resuse old data')

		# get the dataframe from a csv
		df_ = read_csv('data/trend_data/alumni_data_deployment.csv', )
		df_['time'] = to_datetime(df_['time'])
		to_zone = tz.tzlocal()
		df_['time'] = df_['time'].apply(lambda x: x.astimezone(to_zone)) # convert time to loca timezones
		df_.set_index(keys='time',inplace=True, drop = True)
		df_ = a_utils.dropNaNrows(df_)

		# add wet bulb temperature to the data
		log.info('Deploy Control Module: Start of Wet Bulb Data Calculation')
		rh = df_['WeatherDataProfile humidity']/100
		rh = rh.to_numpy()
		t_db = 5*(df_['AHU_1 outdoorAirTemp']-32)/9 + 273.15
		t_db = t_db.to_numpy()
		T = HAPropsSI('T_wb','R',rh,'T',t_db,'P',101325)
		t_f = 9*(T-273.15)/5 + 32
		df_['wbt'] = t_f
		log.info('Deploy Control Module: Wet Bulb Data Calculated')

		# rename the columns
		new_names = []
		for i in df_.columns:
			new_names.append(meta_data_["reverse_col_alias"][i])
		df_.columns = new_names

		# collect current set point
		hist_stpt = df_.loc[df_.index[-1],['sat_stpt']].to_numpy().copy().flatten()

		# clean the data
		df_cleaned = dp.online_data_clean(
			meta_data_ = meta_data_, df = df_
		)

		# clip less than 0 values
		df_cleaned.clip(lower=0, inplace=True)

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

		# also need an unscaled version of the observation for logging
		df_unscaled = df_cleaned.copy()

		# scale the columns: here we will use min-max
		df_cleaned[df_cleaned.columns] = scaler.minmax_scale(df_cleaned, df_cleaned.columns, df_cleaned.columns)

		# create avg_stpt column
		stpt_cols = [ele for ele in df_cleaned.columns if 'vrf' in ele]
		df_cleaned['avg_stpt'] = df_cleaned[stpt_cols].mean(axis=1)
		# drop individual set point cols
		df_cleaned.drop( columns = stpt_cols, inplace = True)
		# rearrange observation cols
		df_cleaned = df_cleaned[obs_space_vars]

		# create avg_stpt column
		stpt_cols = [ele for ele in df_unscaled.columns if 'vrf' in ele]
		df_unscaled['avg_stpt'] = df_unscaled[stpt_cols].mean(axis=1)
		# drop individual set point cols
		df_unscaled.drop( columns = stpt_cols, inplace = True)
		# rearrange observation cols
		df_unscaled = df_unscaled[obs_space_vars] 

		return df_cleaned, df_unscaled, hist_stpt

	except Exception as e:
		log.error('Deploy Control Module: %s', str(e))
		log.debug(e, exc_info=True)




