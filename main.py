"""
This script implements the different threads that are going to be executed for the Alumni Hall deployment.
This script will control the "Supply Air Set Point" for the Air Handling Units. The skeleton script 
is now being created to formalize the main components needed in creating the relearning agent.
"""

from multiprocessing import Event, Lock
from threading import Thread
from datetime import datetime

import warnings
with warnings.catch_warnings():
	from alumni_scripts import data_generator as datagen
	from alumni_scripts import model_learn as mdlearn

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
		'save_path': 'temp/', 'name': 'cwe', 'epoch' : 2000
	}
	exp_params['hwe_model_config'] = {
		'model_type': 'regresion', 'train_batchsize' : 32,
		'input_timesteps': 1, 'input_dim': 4, 'timegap': 6,
		'dense_layers' : 4, 'dense_units': 8, 'activation_dense' : 'relu',
		'lstm_layers' : 4, 'lstm_units': 8, 'activation_lstm' : 'relu',
		'save_path': 'temp/', 'name': 'hwe', 'epoch' : 2000
	}
	exp_params['vlv_model_config'] = {
		'model_type': 'regresion', 'train_batchsize' : 32,
		'input_timesteps': 1, 'input_dim': 4, 'timegap': 6,
		'dense_layers' : 4, 'dense_units': 8, 'activation_dense' : 'relu',
		'lstm_layers' : 4, 'lstm_units': 8, 'activation_lstm' : 'relu',
		'save_path': 'temp/', 'name': 'vlv', 'epoch' : 2000
	}

	lstm_data_available = Event()
	end_learning = Event()
	lstm_train_data_lock = Lock()
	lstm_weights_lock = Lock()
	
	# both of the two below is obtained from meta_data.json
	# agg = 
	# scaler = 

	# cwe_vars = []
	# hwe_vars = []
	# vlv_vars = []

	time_stamp = datetime(year = 2018, month = 11, day = 1, hour=0, minute=0, second=0)

	data_gen_th = Thread(target=datagen.offline_data_gen, daemon = False,
						kwargs={'time_stamp':time_stamp,
						'lstm_data_available':lstm_data_available,
						'end_learning':end_learning,
						'lstm_train_data_lock':lstm_train_data_lock,
						'lstm_weights_lock':lstm_weights_lock})

	# How to set end_learning

