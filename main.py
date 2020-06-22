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
	from alumni_scripts import alumni_data_utils as a_utils

def controller_learn():
	"""
	Learn the controller on a data-driven simulator of the environment
	"""

	raise NotImplementedError


def data_processing():
	"""
	Creating a data base which may be queried to get relevant data
	"""

	raise NotImplementedError


def control_loop():
	"""
	Generates the actual actions to be sent to the actual building
	"""

	raise NotImplementedError

if __name__ == "__main__":
	
	exp_params = {}

	exp_params['cwe_model_config'] = {
		'model_type': 'regresion', 'train_batchsize' : 32,
		'input_timesteps': 1, 'input_dim': 4, 'timegap': 6,
		'dense_layers' : 4, 'dense_units': 8, 'activation_dense' : 'relu',
		'lstm_layers' : 4, 'lstm_units': 8, 'activation_lstm' : 'relu',
		'save_path': 'temp/', 'name': 'cwe', 'epochs' : 100
	}
	exp_params['hwe_model_config'] = {
		'model_type': 'regresion', 'train_batchsize' : 32,
		'input_timesteps': 1, 'input_dim': 4, 'timegap': 6,
		'dense_layers' : 4, 'dense_units': 8, 'activation_dense' : 'relu',
		'lstm_layers' : 4, 'lstm_units': 8, 'activation_lstm' : 'relu',
		'save_path': 'temp/', 'name': 'hwe', 'epochs' : 50
	}
	exp_params['vlv_model_config'] = {
		'model_type': 'classification', 'train_batchsize' : 32,
		'input_timesteps': 1, 'input_dim': 4, 'timegap': 6,
		'dense_layers' : 4, 'dense_units': 8, 'activation_dense' : 'relu',
		'lstm_layers' : 4, 'lstm_units': 8, 'activation_lstm' : 'relu',
		'save_path': 'temp/', 'name': 'vlv', 'epochs' : 100
	}

	lstm_data_available = Event()
	end_learning = Event()
	env_data_available = Event()
	lstm_train_data_lock = Lock()
	lstm_weights_lock = Lock()
	env_train_data_lock = Lock()
	
	# both of the two below is obtained from meta_data.json
	with open('alumni_scripts/meta_data.json', 'r') as fp:
		meta_data_ = json.load(fp)
	agg = meta_data_['column_agg_type']
	scaler = a_utils.dataframescaler(meta_data_['column_stats_half_hour'])

	cwe_vars = ['pchw_flow', 'oah', 'wbt',  'sat', 'oat', 'cwe']
	hwe_vars = ['oat', 'oah', 'wbt', 'sat', 'hwe']
	vlv_vars = ['oat', 'oah', 'wbt', 'sat', 'hwe']
	# how to set prediction sections
	relearn_interval_days = 7

	time_stamp = datetime(year = 2018, month = 11, day = 7, hour=0, minute=0, second=0)

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
								'measurement':'alumni_data_v2'})
	data_gen_th.start()

	model_learn_th = Thread(target=mdlearn.data_driven_model_learn, daemon = False,
						kwargs={
							'lstm_data_available':lstm_data_available,
							'end_learning':end_learning,
							'lstm_train_data_lock':lstm_train_data_lock,
							'lstm_weights_lock':lstm_weights_lock,
							'cwe_model_config':exp_params['cwe_model_config'],
							'hwe_model_config':exp_params['hwe_model_config'],
							'vlv_model_config':exp_params['vlv_model_config'],

						}
	)
	model_learn_th.start()

	model_learn_th.join()
	data_gen_th.join()
	print("End of program execution")
	# How to set end_learning

