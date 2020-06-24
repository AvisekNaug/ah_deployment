"""
This script will have methods/Threads that help create data for different other threads
1. offline_data_gen:
	Whenever lstm_train_data_lock, env_train_data_lock is free, do in a never-ending loop
	a. read the stored file and create lstm related data for the last 3 months + 1 week
	b. read the stored file and create environment related data for last 3 months + 1 week over same period
	* This method needs to know a timestamp as an input based on which it will get data from offline database.
2. online_data_process : Use stats generated by data_stats to process data online
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

# imports
import numpy as np
import json
from datetime import datetime, timedelta
from influxdb import DataFrameClient
from threading import Thread
from multiprocessing import Event, Lock

from alumni_scripts import alumni_data_utils as a_utils

from keras.utils import to_categorical


def offline_data_gen(*args, **kwargs):

	time_stamp = kwargs['time_stamp']
	# year and week
	year_num, week_num, _ = time_stamp.isocalendar()

	# Events
	lstm_data_available : Event = kwargs['lstm_data_available']  # new data available for lstm relearning
	env_data_available : Event = kwargs['env_data_available']  # new data available for env relearning  # pylint: disable=unused-variable
	end_learning : Event = kwargs['end_learning'] 

	# Locks
	lstm_train_data_lock : Lock = kwargs['lstm_train_data_lock']  # prevent dataloop from writing data
	env_train_data_lock : Lock = kwargs['env_train_data_lock']  # prevent dataloop from writing data  # pylint: disable=unused-variable


	client = DataFrameClient(host='localhost', port=8086, database=kwargs['database'],)


	while True:

		"""relearning interval decider: here it is a condition; for online it will be time interval or error measure 
		along side the already exisitng conditions"""
		if not (lstm_data_available.is_set() | env_data_available.is_set()):  # or condition prevents faster overwrite for env data


			result_obj = client.query("select * from {} where time >= '{}' - 13w \
								and time < '{}'".format(kwargs['measurement'], str(time_stamp), str(time_stamp)))
			df_env = result_obj[kwargs['measurement']]
			df_env = df_env.drop(columns = ['data_cleaned', 'aggregated', 'time-interval'])

			data_gen_process_cwe_th = Thread(target=data_gen_process_cwe, daemon=False,
										kwargs={ 
										'df' : result_obj[kwargs['measurement']].loc[:,kwargs['cwe_vars']],
										'agg': kwargs['agg'], 'scaler': kwargs['scaler'], 'year_num': year_num,
										'week_num': week_num, 'save_path':kwargs['save_path'] 
										})
			data_gen_process_hwe_th = Thread(target=data_gen_process_hwe, daemon=False, 
										kwargs={ 
										'df' : result_obj[kwargs['measurement']].loc[:,kwargs['hwe_vars']],
										'agg': kwargs['agg'], 'scaler': kwargs['scaler'], 'year_num': year_num,
										'week_num': week_num, 'save_path':kwargs['save_path'] 
										})
			data_gen_process_vlv_th = Thread(target=data_gen_process_vlv, daemon=False, 
										kwargs={ 
										'df' : result_obj[kwargs['measurement']].loc[:,kwargs['vlv_vars']],
										'agg': kwargs['agg'], 'scaler': kwargs['scaler'], 'year_num': year_num,
										'week_num': week_num, 'save_path':kwargs['save_path'] 
										})
			data_gen_process_env_th = Thread(target=data_gen_process_env, daemon=False, 
										kwargs={
										'df' : df_env,
										'agg': kwargs['agg'], 'scaler': kwargs['scaler'],
										'save_path':kwargs['save_path']
										})


			with lstm_train_data_lock:
				data_gen_process_cwe_th.start()
				data_gen_process_vlv_th.start()
				data_gen_process_cwe_th.join()
				data_gen_process_vlv_th.join()
			
			with env_train_data_lock:
				
				data_gen_process_env_th.start()
				data_gen_process_env_th.join()

			with lstm_train_data_lock:
				data_gen_process_hwe_th.start()
				data_gen_process_hwe_th.join()
			
			#lstm_data_available.set()  # data is now available for lstm training
			#env_data_available.set()  # data is now available for agent and env training
		
			time_stamp += timedelta(days=kwargs['relearn_interval_days'])
			week_num += 1
			week_num = week_num if week_num%53 != 0 else 1
			year_num = year_num if week_num!= 1 else year_num+1

			if week_num == 49:  # can be other terminating condition like year==2020 & week=5 etc
				end_learning.set()
				break


def data_gen_process_cwe(*args, **kwargs):
	
	# read the data from the database
	df = kwargs['df'].copy()


	# smooth the data
	#df = a_utils.dfsmoothing(df=df, column_names=list(df.columns))
	df.clip(lower=0) # Remove <0 values for all columns as a result of smoothing
	

	# aggregate data
	rolling_sum_target, rolling_mean_target = [], []
	for col_name in df.columns:
		if kwargs['agg'][col_name] == 'sum' : rolling_sum_target.append(col_name)
		else: rolling_mean_target.append(col_name)
	
	df[rolling_sum_target] =  a_utils.window_sum(df, window_size=6, column_names=rolling_sum_target)
	df[rolling_mean_target] =  a_utils.window_sum(df, window_size=6, column_names=rolling_mean_target)
	df = a_utils.dropNaNrows(df)


	# Sample the data at period intervals
	df = a_utils.sample_timeseries_df(df, period=6)


	# scale the columns: here we will use min-max
	df[df.columns] = kwargs['scaler'].minmax_scale(df, df.columns, df.columns)


	# creating sat-oat for the data
	df['sat-oat'] = df['sat'] - df['oat']


	# select non-zero operating regions
	df = a_utils.df2operating_regions(df, ['cwe'], 0.0)


	# determine split point for last 1 week test data
	t_train_end = df.index[-1] - timedelta(days=7)
	test_df = df.loc[t_train_end : , : ]
	splitvalue = test_df.shape[0]

	# create train and test/validate data
	X_train, X_test, y_train, y_test = a_utils.df_2_arrays(df = df,
		predictorcols = ['sat-oat', 'oah', 'wbt', 'pchw_flow'], outputcols = ['cwe'], lag = 0,
		scaling = False, scaler = None, scaleX = True, scaleY = True,
		split=splitvalue, shuffle=False,
		reshaping=True, input_timesteps=1, output_timesteps = 1,)


	# save test ids for later plots
	idx_end = -max(X_test.shape[1],y_test.shape[1])
	idx_start = idx_end - X_test.shape[0] + 1
	test_idx = df.index[[ i for i in range(idx_start, idx_end+1, 1) ]]
	test_info = {'test_idx' : [str(i) for i in test_idx], 'year_num': kwargs['year_num'], 'week_num':kwargs['week_num'] }
	with open(kwargs['save_path']+'cwe_data/cwe_test_info.txt', 'a') as ifile:
		ifile.write(json.dumps(test_info)+'\n')      

	np.save(kwargs['save_path']+'cwe_data/cwe_X_train.npy', X_train)
	np.save(kwargs['save_path']+'cwe_data/cwe_X_val.npy', X_test)
	np.save(kwargs['save_path']+'cwe_data/cwe_y_train.npy', y_train)
	np.save(kwargs['save_path']+'cwe_data/cwe_y_val.npy', y_test)


def data_gen_process_hwe(*args, **kwargs):
	
	# try:
	# read the data from the database
	df = kwargs['df'].copy()


	# smooth the data
	# df = a_utils.dfsmoothing(df=df, column_names=list(df.columns))
	df.clip(lower=0) # Remove <0 values for all columns as a result of smoothing
	

	# aggregate data
	rolling_sum_target, rolling_mean_target = [], []
	for col_name in df.columns:
		if kwargs['agg'][col_name] == 'sum' : rolling_sum_target.append(col_name)
		else: rolling_mean_target.append(col_name)
	
	df[rolling_sum_target] =  a_utils.window_sum(df, window_size=6, column_names=rolling_sum_target)
	df[rolling_mean_target] =  a_utils.window_sum(df, window_size=6, column_names=rolling_mean_target)
	df = a_utils.dropNaNrows(df)


	# Sample the data at period intervals
	df = a_utils.sample_timeseries_df(df, period=6)


	# scale the columns: here we will use min-max
	df[df.columns] = kwargs['scaler'].minmax_scale(df, df.columns, df.columns)


	# creating sat-oat for the data
	df['sat-oat'] = df['sat'] - df['oat']


	# select non-zero operating regions
	df = a_utils.df2operating_regions(df, ['hwe'], 0.0)


	# determine split point for last 1 week test data
	t_train_end = df.index[-1] - timedelta(days=7)
	test_df = df.loc[t_train_end : , : ]
	splitvalue = test_df.shape[0]

	# create train and test/validate data
	X_train, X_test, y_train, y_test = a_utils.df_2_arrays(df = df,
		predictorcols = ['oat', 'oah', 'wbt', 'sat-oat'], outputcols = ['hwe'], lag = 0,
		scaling = False, scaler = None, scaleX = True, scaleY = True,
		split=splitvalue, shuffle=False,
		reshaping=True, input_timesteps=1, output_timesteps = 1,)


	# save test ids for later plots
	idx_end = -max(X_test.shape[1],y_test.shape[1])
	idx_start = idx_end - X_test.shape[0] + 1
	test_idx = df.index[[ i for i in range(idx_start, idx_end+1, 1) ]]
	test_info = {'test_idx' : [str(i) for i in test_idx], 'year_num': kwargs['year_num'], 'week_num':kwargs['week_num'] }
	with open(kwargs['save_path']+'hwe_data/hwe_test_info.txt', 'a') as ifile:
		ifile.write(json.dumps(test_info)+'\n')      

	np.save(kwargs['save_path']+'hwe_data/hwe_X_train.npy', X_train)
	np.save(kwargs['save_path']+'hwe_data/hwe_X_val.npy', X_test)
	np.save(kwargs['save_path']+'hwe_data/hwe_y_train.npy', y_train)
	np.save(kwargs['save_path']+'hwe_data/hwe_y_val.npy', y_test)
	
	# except Exception:
	# 	import traceback
	# 	print(traceback.format_exc())


def data_gen_process_vlv(*args, **kwargs):
	
	# read the data from the database
	df = kwargs['df'].copy()


	# smooth the data
	# df = a_utils.dfsmoothing(df=df, column_names=list(df.columns))
	df.clip(lower=0) # Remove <0 values for all columns as a result of smoothing
	

	# aggregate data
	rolling_sum_target, rolling_mean_target = [], []
	for col_name in df.columns:
		if kwargs['agg'][col_name] == 'sum' : rolling_sum_target.append(col_name)
		else: rolling_mean_target.append(col_name)
	
	df[rolling_sum_target] =  a_utils.window_sum(df, window_size=6, column_names=rolling_sum_target)
	df[rolling_mean_target] =  a_utils.window_sum(df, window_size=6, column_names=rolling_mean_target)
	df = a_utils.dropNaNrows(df)


	# Sample the data at period intervals
	df = a_utils.sample_timeseries_df(df, period=6)


	# scale the columns: here we will use min-max
	df[df.columns] = kwargs['scaler'].minmax_scale(df, df.columns, df.columns)


	# creating sat-oat for the data
	df['sat-oat'] = df['sat'] - df['oat']


	# add binary classification column
	df['vlv'] = 1.0
	df.loc[df['hwe']<= 0.0, ['vlv']] = 0


	# determine split point for last 1 week test data
	t_train_end = df.index[-1] - timedelta(days=7)
	test_df = df.loc[t_train_end : , : ]
	splitvalue = test_df.shape[0]

	# create train and test/validate data
	X_train, X_test, y_train, y_test = a_utils.df_2_arrays(df = df,
		predictorcols = ['oat', 'oah', 'wbt', 'sat-oat'], outputcols = ['vlv'], lag = 0,
		scaling = False, scaler = None, scaleX = True, scaleY = True,
		split=splitvalue, shuffle=False,
		reshaping=True, input_timesteps=1, output_timesteps = 1,)

	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)


	# save test ids for later plots
	idx_end = -max(X_test.shape[1],y_test.shape[1])
	idx_start = idx_end - X_test.shape[0] + 1
	test_idx = df.index[[ i for i in range(idx_start, idx_end+1, 1) ]]
	test_info = {'test_idx' : [str(i) for i in test_idx], 'year_num': kwargs['year_num'], 'week_num':kwargs['week_num'] }
	with open(kwargs['save_path']+'vlv_data/vlv_test_info.txt', 'a') as ifile:
		ifile.write(json.dumps(test_info)+'\n')      

	np.save(kwargs['save_path']+'vlv_data/vlv_X_train.npy', X_train)
	np.save(kwargs['save_path']+'vlv_data/vlv_X_val.npy', X_test)
	np.save(kwargs['save_path']+'vlv_data/vlv_y_train.npy', y_train)
	np.save(kwargs['save_path']+'vlv_data/vlv_y_val.npy', y_test)


def data_gen_process_env(*args, **kwargs):

	# read the data from the database
	df = kwargs['df'].copy()

	# smooth the data
	# df = a_utils.dfsmoothing(df=df, column_names=list(df.columns))
	df.clip(lower=0) # Remove <0 values for all columns as a result of smoothing
	

	# aggregate data
	rolling_sum_target, rolling_mean_target = [], []
	for col_name in df.columns:
		if kwargs['agg'][col_name] == 'sum' : rolling_sum_target.append(col_name)
		else: rolling_mean_target.append(col_name)
	
	df[rolling_sum_target] =  a_utils.window_sum(df, window_size=6, column_names=rolling_sum_target)
	df[rolling_mean_target] =  a_utils.window_sum(df, window_size=6, column_names=rolling_mean_target)
	df = a_utils.dropNaNrows(df)


	# Sample the data at period intervals
	df = a_utils.sample_timeseries_df(df, period=6)


	# scale the columns: here we will use min-max
	df[df.columns] = kwargs['scaler'].minmax_scale(df, df.columns, df.columns)

	# creating sat-oat for the data
	df['sat-oat'] = df['sat'] - df['oat']

	# create avg_stpt column
	stpt_cols = [ele for ele in df.columns if 'vrf' in ele]
	df['avg_stpt'] = df[stpt_cols].mean(axis=1)
	# drop individual set point cols
	df.drop( columns = stpt_cols, inplace = True)

	# save the data frame
	df.to_pickle(kwargs['save_path']+'env_data/env_data.pkl')


def online_data_gen(*args, **kwargs):
	"""
	This process should trigger data collection and correspondingly trigger other lstm and agent
	relearning modules based on current time being at the end of a regular interval of whatever(or error
	tracking measure). Also, it should use the bdx api to get the raw data-> clean and process it 
	using data_process -> and finally do the data_gen_* methods which will trigger model learning
	which is followed by model training
	"""
	raise NotImplementedError