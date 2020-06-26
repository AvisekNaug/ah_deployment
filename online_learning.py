"""
This script implements the different threads that are going to be executed for the Alumni Hall deployment.
This script will control the "Supply Air Set Point" for the Air Handling Units. The skeleton script 
is now being created to formalize the main components needed in creating the relearning agent.
"""
import os
# Enable '0' or disable '' GPU use
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from multiprocessing import Event, Lock
from threading import Thread

from datetime import datetime
import json

import warnings
with warnings.catch_warnings():

	from alumni_scripts import data_generator as datagen
	from alumni_scripts import model_learn as mdlearn
	from alumni_scripts import control_learn as ctlearn

	from alumni_scripts import alumni_data_utils as a_utils
	from source import utils

if __name__ == "__main__":
	
	exp_params = {}

	# how to set prediction sections
	relearn_interval_days = 7
	# number of epochs to train dynamic models
	epochs = 100
	# num of steps to learn rl in each train method
	rl_train_steps = 6000
	# time stamp of the last time point in the test data
	time_stamp = datetime(year = 2018, month = 11, day = 7, hour=0, minute=0, second=0)

	save_path = 'tmp/'
	model_path = 'models/'
	log_path = 'logs/'
	results = 'results/'
	trend_data = 'data/trend_data/'
	cwe_data = save_path + 'cwe_data/'
	hwe_data = save_path + 'hwe_data/'
	vlv_data = save_path + 'vlv_data/'
	env_data = save_path + 'env_data/'
	rl_perf_data = save_path + 'rl_perf_data/'

	utils.make_dirs(cwe_data)
	utils.make_dirs(hwe_data)
	utils.make_dirs(vlv_data)
	utils.make_dirs(env_data)
	utils.make_dirs(model_path)
	utils.make_dirs(log_path)
	utils.make_dirs(results)
	utils.make_dirs(rl_perf_data)
	utils.make_dirs(trend_data)

	exp_params['cwe_model_config'] = {
		'model_type': 'regresion', 'train_batchsize' : 32,
		'input_timesteps': 1, 'input_dim': 4, 'timegap': 6,
		'dense_layers' : 4, 'dense_units': 8, 'activation_dense' : 'relu',
		'lstm_layers' : 4, 'lstm_units': 8, 'activation_lstm' : 'relu',
		'save_path': cwe_data, 'model_path': model_path, 'name': 'cwe', 'epochs' : epochs
	}
	cwe_vars = ['pchw_flow', 'oah', 'wbt',  'sat', 'oat', 'cwe']

	exp_params['hwe_model_config'] = {
		'model_type': 'regresion', 'train_batchsize' : 32,
		'input_timesteps': 1, 'input_dim': 4, 'timegap': 6,
		'dense_layers' : 4, 'dense_units': 8, 'activation_dense' : 'relu',
		'lstm_layers' : 4, 'lstm_units': 8, 'activation_lstm' : 'relu',
		'save_path': hwe_data, 'model_path': model_path, 'name': 'hwe', 'epochs' : epochs
	}
	hwe_vars = ['oat', 'oah', 'wbt', 'sat', 'hwe']
	
	exp_params['vlv_model_config'] = {
		'model_type': 'classification', 'train_batchsize' : 32,
		'input_timesteps': 1, 'input_dim': 4, 'timegap': 6,
		'dense_layers' : 4, 'dense_units': 8, 'activation_dense' : 'relu',
		'lstm_layers' : 4, 'lstm_units': 8, 'activation_lstm' : 'relu',
		'save_path': vlv_data, 'model_path': model_path, 'name': 'vlv', 'epochs' : epochs
	}
	vlv_vars = ['oat', 'oah', 'wbt', 'sat', 'hwe']

	exp_params['env_config'] = {
		'save_path' : env_data, 'model_path': model_path, 'logs' : log_path,
		'obs_space_vars' : ['oat', 'oah', 'wbt', 'avg_stpt', 'sat'], 
		'action_space_vars' :['sat'], 
		'cwe_inputs' : ['sat-oat', 'oah', 'wbt', 'pchw_flow'],
		'hwe_inputs' : ['oat', 'oah', 'wbt', 'sat-oat'],
		'vlv_inputs' : ['oat', 'oah', 'wbt', 'sat-oat'],
	}

	# Events
	lstm_data_available = Event()  # new data available for lstm relearning
	end_learning = Event()  # end the relearning procedure
	env_data_available = Event()  # new data available for alumni env rl learning
	lstm_weights_available = Event()  # trained lstm models are avilable
	agent_model_available = Event()  # trained controller weights are availalbe for "online" deployment
	agent_weights_available = Event()  # agent weights are available to be read by deploy loop
	agent_weights_available.set() # set agent weights to available in online learning as it
	# will be immediately depoloyed

	# Locks
	lstm_train_data_lock = Lock()  # read / write lstm data without access issues
	lstm_weights_lock = Lock()  # read / write lstm weights without access issues
	env_train_data_lock = Lock()  # read / write env data without access issues
	agent_weights_lock = Lock()  #  read / write rl weights without access issues
	
	# get agg type and data stats from meta_data.json
	with open('alumni_scripts/meta_data.json', 'r') as fp:
		meta_data_ = json.load(fp)
	agg = meta_data_['column_agg_type']
	scaler = a_utils.dataframescaler(meta_data_['column_stats_half_hour'])

	data_gen_th = Thread(target=datagen.offline_data_gen, daemon = False,
						kwargs={'time_stamp':time_stamp,
								'lstm_data_available':lstm_data_available,
								'end_learning':end_learning,
								'lstm_train_data_lock':lstm_train_data_lock,
								'lstm_weights_lock':lstm_weights_lock,
								'relearn_interval_days':relearn_interval_days,
								'env_data_available':env_data_available,
								'env_train_data_lock':env_train_data_lock,
								'agg' : agg,
								'scaler' : scaler,
								'cwe_vars': cwe_vars,
								'hwe_vars': hwe_vars,
								'vlv_vars': vlv_vars,
								'database':'bdx_batch_db',
								'measurement':'alumni_data_v2',
								'save_path': save_path})
	data_gen_th.start()

	model_learn_th = Thread(target=mdlearn.data_driven_model_learn, daemon = False,
						kwargs={'lstm_data_available':lstm_data_available,
								'end_learning':end_learning,
								'lstm_train_data_lock':lstm_train_data_lock,
								'lstm_weights_lock':lstm_weights_lock,
								'lstm_weights_available':lstm_weights_available,
								'cwe_model_config':exp_params['cwe_model_config'],
								'hwe_model_config':exp_params['hwe_model_config'],
								'vlv_model_config':exp_params['vlv_model_config'],
								'save_path': save_path,
								})
	model_learn_th.start()

	ctrl_learn_th = Thread(target=ctlearn.controller_learn, daemon = False,
						kwargs={
							'env_config':exp_params['env_config'],
							'env_data_available' : env_data_available,
							'lstm_weights_available' : lstm_weights_available,
							'agent_weights_available' : agent_weights_available,
							'end_learning':end_learning,
							'env_train_data_lock' : env_train_data_lock,
							'lstm_weights_lock' : lstm_weights_lock,
							'agent_weights_lock' : agent_weights_lock,
							'rl_train_steps' : rl_train_steps,
							'rl_perf_data' : rl_perf_data,
						}

	)
	ctrl_learn_th.start()

	ctrl_learn_th.join()
	model_learn_th.join()
	data_gen_th.join()
	print("End of program execution")

