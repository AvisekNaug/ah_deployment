"""
This script will learn the different data driven models for Alumni Hall
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
from threading import Thread
from multiprocessing import Event, Lock

from source import datamodels as dm

def data_driven_model_learn(*args, **kwargs):
	"""
	Learn the data driven models needed for creating the data-driven 
	simulator of the environment
	"""
	# logger
	log = kwargs['logger']
	try:
		# Events
		lstm_data_available : Event = kwargs['lstm_data_available']  # new data available for relearning
		end_learning : Event = kwargs['end_learning']  # to break out of non-stoping learning offline
		lstm_weights_available : Event = kwargs['lstm_weights_available']
		# Locks
		lstm_train_data_lock : Lock = kwargs['lstm_train_data_lock']  # prevent dataloop from writing data
		lstm_weights_lock : Lock = kwargs['lstm_weights_lock']  # prevent other loops from reading LSTM weights
		# check variables
		models_created = False
		eval_interval = 1
		# to_break = False
		# user validation loss or not
		if kwargs['use_val']:
			cwe_type,hwe_type,vlv_type  = dm.nn_model, dm.nn_model, dm.nn_model
		else:
			cwe_type,hwe_type,vlv_type  = dm.no_val_nn_model, dm.no_val_nn_model, dm.no_val_nn_model

		# Read the processed data and learn the 3 models inside a conditional loop
		while (not end_learning.is_set()) | (lstm_data_available.is_set()):

			if not models_created:  # create the 3 models needed for training

				cwe_model = cwe_type(**kwargs['cwe_model_config'])
				cwe_model.design(**kwargs['cwe_model_config'])
				cwe_model.compile()
				hwe_model = hwe_type(**kwargs['hwe_model_config'])
				hwe_model.design(**kwargs['hwe_model_config'])
				hwe_model.compile()
				vlv_model = vlv_type(**kwargs['vlv_model_config'])
				vlv_model.design(**kwargs['vlv_model_config'])
				vlv_model.compile()
				log.info("Dynamic Model Learning Module: Models Initialized")

				# if models are available from previous offline training
				if lstm_weights_available.is_set():
					with lstm_weights_lock:
						cwe_model.load_weights()
						hwe_model.load_weights()
						vlv_model.load_weights()
					lstm_weights_available.clear()
					log.info("Dynamic Model Learning Module: Models Loaded from Offline Phase")
				models_created = True
				log.info

			# data is available and prev models have been read by env
			if (lstm_data_available.is_set() & (not lstm_weights_available.is_set())): 
				log.info("Dynamic Model Learning Module: Model Training Starts")
				""" Read the train and eval data """
				with lstm_train_data_lock:
					X_train_cwe, y_train_cwe, X_val_cwe, y_val_cwe = np.load(kwargs['save_path']+'cwe_data/cwe_X_train.npy'),\
						np.load(kwargs['save_path']+'cwe_data/cwe_y_train.npy'), np.load(kwargs['save_path']+'cwe_data/cwe_X_val.npy'), \
						np.load(kwargs['save_path']+'cwe_data/cwe_y_val.npy')
					X_train_hwe, y_train_hwe, X_val_hwe, y_val_hwe = np.load(kwargs['save_path']+'hwe_data/hwe_X_train.npy'),\
						np.load(kwargs['save_path']+'hwe_data/hwe_y_train.npy'), np.load(kwargs['save_path']+'hwe_data/hwe_X_val.npy'), \
						np.load(kwargs['save_path']+'hwe_data/hwe_y_val.npy')
					X_train_vlv, y_train_vlv, X_val_vlv, y_val_vlv = np.load(kwargs['save_path']+'vlv_data/vlv_X_train.npy'),\
						np.load(kwargs['save_path']+'vlv_data/vlv_y_train.npy'), np.load(kwargs['save_path']+'vlv_data/vlv_X_val.npy'), \
						np.load(kwargs['save_path']+'vlv_data/vlv_y_val.npy')
				lstm_data_available.clear()

				""" Begin the training """
				with lstm_weights_lock:  # don't let other threds read weights while training in session
					th_cwe_learn = Thread(target=cwe_model.fit, daemon=False, kwargs=
											{'X_train':X_train_cwe,'y_train':y_train_cwe,
											'X_val':X_val_cwe,    'y_val':y_val_cwe,
											'epochs':kwargs['cwe_model_config']['epochs']})
					th_cwe_learn.start()
					th_hwe_learn = Thread(target=hwe_model.fit, daemon=False, kwargs=
											{'X_train':X_train_hwe,'y_train':y_train_hwe,
											'X_val':X_val_hwe,    'y_val':y_val_hwe,
											'epochs':kwargs['hwe_model_config']['epochs'],})
					th_hwe_learn.start()
					th_vlv_learn = Thread(target=vlv_model.fit, daemon=False, kwargs=
											{'X_train':X_train_vlv,'y_train':y_train_vlv,
											'X_val':X_val_vlv,    'y_val':y_val_vlv,
											'epochs':kwargs['vlv_model_config']['epochs'],})
					th_vlv_learn.start()
					# wait for all threads to finish before looping again
					th_cwe_learn.join()
					th_hwe_learn.join()
					th_vlv_learn.join()
					# weights are available
					lstm_weights_available.set()
					log.info("Dynamic Model Learning Module: Model Training Finished")

				"""Prediction"""
				# predict on test data cwe
				cwe_prediction, cwe_target = cwe_model.predict(**{'X_test':X_val_cwe}).flatten(), y_val_cwe.flatten()
				# save the output
				np.save(kwargs['save_path']+'cwe_data/cwe_prediction_interval_{}.npy'.format(eval_interval), cwe_prediction)
				np.save(kwargs['save_path']+'cwe_data/cwe_target_interval_{}.npy'.format(eval_interval), cwe_target)
				# predict on test data hwe
				hwe_prediction, hwe_target = hwe_model.predict(**{'X_test':X_val_hwe}).flatten(), y_val_hwe.flatten()
				# save the output
				np.save(kwargs['save_path']+'hwe_data/hwe_prediction_interval_{}.npy'.format(eval_interval), hwe_prediction)
				np.save(kwargs['save_path']+'hwe_data/hwe_target_interval_{}.npy'.format(eval_interval), hwe_target)
				# predict on test data vlv
				vlv_prediction, vlv_target = vlv_model.predict(**{'X_test':X_val_vlv}), y_val_hwe
				# save the output
				np.save(kwargs['save_path']+'vlv_data/vlv_pred_interval_{}.npy'.format(eval_interval), vlv_prediction)
				np.save(kwargs['save_path']+'vlv_data/vlv_target_interval_{}.npy'.format(eval_interval), vlv_target)
				eval_interval += 1
				log.info("Dynamic Model Learning Module: Model Prediction Finished")

				"""re-init lstm certanin layers"""
				cwe_model.re_init_layers()
				hwe_model.re_init_layers()
				vlv_model.re_init_layers()
				log.info("Dynamic Model Learning Module: Model LSTM Layers Re-initialized")
				
				# if no more learning is needed end this thread
				# if to_break:
				# 	break
				# if end_learning.is_set():  # break after the next loop
				# 	to_break = True
	except Exception as e:
		log.error('Dynamic Model Learning Module: %s', str(e))
		log.debug(e, exc_info=True)
