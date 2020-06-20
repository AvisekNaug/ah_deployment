"""
This script will learn the different data driven models for Alumni Hall
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
from threading import Thread


from source import datamodels as dm

def data_driven_model_learn(*args, **kwargs):
	"""
	Learn the data driven models needed for creating the data-driven 
	simulator of the environment
	"""

	# Events
	lstm_data_available = kwargs['lstm_data_available']  # new data available for relearning
	end_learning = kwargs['end_learning']  # to break out of non-stoping learning offline
	# Locks
	lstm_train_data_lock = kwargs['lstm_data_read_lock']  # prevent dataloop from writing data
	lstm_weights_lock = kwargs['lstm_weights_lock']  # prevent other loops from reading LSTM weights
	# check variables
	models_created = False

	# Read the processed data and learn the 3 models inside a conditional loop
	while True:

		# if no more learning is needed end this thread
		if end_learning.is_set():
			end_learning.clear()
			break

		if not models_created:  # create the 3 models needed for training

			cwe_model = dm.nn_model(*args, kwargs['cwe_model_config'])
			cwe_model.design(*args, kwargs['cwe_model_config'])
			hwe_model = dm.nn_model(*args, kwargs['hwe_model_config'])
			hwe_model.design(*args, kwargs['hwe_model_config'])
			vlv_model = dm.nn_model(*args, kwargs['vlv_model_config'])
			vlv_model.design(*args, kwargs['vlv_model_config'])

			models_created = True

		if lstm_data_available.is_set():  # data is available; start training each model in parallel

			# TODO: Have to implement this in parallel --> simple start threads here and join them
			# These threads run the fit function for each model class

			# read train and val data for cwe,hwe and vlv models
			with lstm_train_data_lock:
				X_train_cwe, y_train_cwe, X_val_cwe, y_val_cwe = np.load('temp/X_train_cwe.npy'),\
				np.load('temp/y_train_cwe.npy'), np.load('temp/X_val_cwe.npy'), np.load('temp/y_val_cwe.npy')
				X_train_hwe, y_train_hwe, X_val_hwe, y_val_hwe = np.load('temp/X_train_hwe.npy'),\
				np.load('temp/y_train_hwe.npy'), np.load('temp/X_val_hwe.npy'), np.load('temp/y_val_hwe.npy')
				X_train_vlv, y_train_vlv, X_val_vlv, y_val_vlv = np.load('temp/X_train_vlv.npy'),\
				np.load('temp/y_train_vlv.npy'), np.load('temp/X_val_vlv.npy'), np.load('temp/y_val_vlv.npy')
			lstm_data_available.clear()

			with lstm_weights_lock:  # don't let other threds read weights while training in session
				th_cwe_learn = Thread(target=cwe_model.fit, daemon=False, kwargs=
										{'X_train':X_train_cwe,'y_train':y_train_cwe,
										'X_val':X_val_cwe,    'y_val':y_val_cwe,
										'epochs':kwargs['epochs']})
				th_cwe_learn.start()
				th_hwe_learn = Thread(target=hwe_model.fit, daemon=False, kwargs=
										{'X_train':X_train_hwe,'y_train':y_train_hwe,
										'X_val':X_val_hwe,    'y_val':y_val_hwe,
										'epochs':kwargs['epochs'],})
				th_hwe_learn.start()
				th_vlv_learn = Thread(target=vlv_model.fit, daemon=False, kwargs=
										{'X_train':X_train_vlv,'y_train':y_train_vlv,
										'X_val':X_val_vlv,    'y_val':y_val_vlv,
										'epochs':kwargs['epochs'],})
				th_vlv_learn.start()
				# wait for all threads to finish before looping again
				th_cwe_learn.join()
				th_hwe_learn.join()
				th_vlv_learn.join()

