"""
This scripts contains methods that run learn the rl agent for control
"""
import numpy as np

import warnings
with warnings.catch_warnings():
	from stable_baselines.common.vec_env import DummyVecEnv
	from stable_baselines.common import set_global_seeds, make_vec_env
	from keras.models import load_model
	from alumni_scripts import alumni_env
	from alumni_scripts import ppo_agent
	from alumni_scripts import alumni_env_wrapper
	from alumni_scripts.alumni_data_utils import rl_perf_save

from pandas import read_pickle
from multiprocessing import Event, Lock


def controller_learn(*args, **kwargs):
	"""
	Learn the controller on a data-driven simulator of the environment
	"""
	env_data_available : Event = kwargs['env_data_available']  # new data available for env relearning
	lstm_weights_available : Event = kwargs['lstm_weights_available']  # lstm weights are available for alumni env
	agent_weights_available : Event = kwargs['agent_weights_available']  # deploy loop can read the agent weights now
	end_learning : Event = kwargs['end_learning']  # to break out of non-stoping learning offline

	env_train_data_lock : Lock = kwargs['env_train_data_lock']  # prevent data read/write access
	lstm_weights_lock : Lock = kwargs['lstm_weights_lock']  # prevent data read/write access
	agent_weights_lock : Lock = kwargs['agent_weights_lock']  # prevent data read/write access
	# check variables
	interval = 1
	env_created = False
	agent_created = False
	writeheader = True
	to_break = False

	while True:

		# if data for alumni env and energy models are available, and agent_weights have not been updated
		# after last read then run controller training --!! To ber set to clear for offline training
		if ( env_data_available.is_set() & lstm_weights_available.is_set() & (not agent_weights_available.is_set()) ):

			print("******Entering Control Learn Loop*******")

			with env_train_data_lock:
				df_scaled = read_pickle(kwargs['env_config']['save_path']+'env_data.pkl')
			df_scaled_stats = df_scaled.describe()
			env_data_available.clear()

			with lstm_weights_lock:
				cwe_energy_model  =load_model(kwargs['env_config']['model_path']+'cwe_best_model')
				hwe_energy_model  =load_model(kwargs['env_config']['model_path']+'hwe_best_model')
				vlv_energy_model  =load_model(kwargs['env_config']['model_path']+'vlv_best_model')
			lstm_weights_available.clear()

			"""create environment with new data"""
			monitor_dir = kwargs['env_config']['logs']+'Interval_{}/'.format(interval)

			
			"""Arguments to be fed to the custom environment inside make_vec_env"""
			reward_params = {'energy_saved': 100.0, 'energy_savings_thresh': 0.0,
							 'energy_penalty': -100.0, 'energy_reward_weight': 0.5,
							 'comfort': 1.0, 'comfort_thresh': 0.10,
							 'uncomfortable': -1.0, 'comfort_reward_weight': 0.5,
							 'action_minmax':[np.array([72]), np.array([65])]
							 }
			env_kwargs = dict(  #  Optional keyword argument to pass to the env constructor
				df = df_scaled,
				totaldf_stats = df_scaled_stats,
				obs_space_vars=kwargs['env_config']['obs_space_vars'],
				action_space_vars=kwargs['env_config']['action_space_vars'],
				action_space_bounds=[[-4.0], [4.0]],  # bounds for real world action space; is scaled
				# internally using the reward_params; the agent weights output action in this +-4 range

				cwe_energy_model=cwe_energy_model,
				cwe_input_vars=kwargs['env_config']['cwe_inputs'],
				cwe_input_shape=(1, 1, len(kwargs['env_config']['cwe_inputs'])),

				hwe_energy_model=hwe_energy_model,
				hwe_input_vars=kwargs['env_config']['hwe_inputs'],
				hwe_input_shape=(1, 1, len(kwargs['env_config']['hwe_inputs'])),

				vlv_state_model=vlv_energy_model,
				vlv_input_vars=kwargs['env_config']['vlv_inputs'],
				vlv_input_shape=(1, 1, len(kwargs['env_config']['vlv_inputs'])),

				**reward_params  # the reward adjustment parameters
			)

			env_id = alumni_env.Env  # the environment ID or the environment class
			start_index = 0  # start rank index
			vec_env_cls = DummyVecEnv

			""" create the Gym env"""
			if not env_created:
				env = alumni_env_wrapper.custom_make_vec_env(
					env_id=env_id, n_envs=1, start_index = start_index, monitor_dir = monitor_dir,
					vec_env_cls = vec_env_cls, env_kwargs = env_kwargs
				)
				env_created = True
			else:  # re-initialize env with new interval's data
				# change the monitor log directory
				env.env_method('changelogpath', (monitor_dir))
				# reinitialize the environment with new data and models
				env.env_method('re_init_env', **env_kwargs)

			"""provide envrionment to the new rl agent for ppo to decide its state and actions spaces"""
			if not agent_created:
				agent = ppo_agent.get_agent(env=env, 
											model_save_dir=kwargs['env_config']['model_path'],
											monitor_log_dir = kwargs['env_config']['logs'])
				agent_created = True
			# ** agent uses "monitor_log_dir" to update agent by looking at rewards
			
			"""Start training the agent"""
			env.env_method('trainenv')
			with agent_weights_lock:
				agent = ppo_agent.train_agent(agent, env = env, steps=kwargs['rl_train_steps'])

			"""test rl model"""
			env.env_method('testenv')
			# provide path to the current best rl agent weights and test it
			best_rl_agent_path = kwargs['env_config']['model_path'] + 'best_rl_agent'
			with agent_weights_lock:
				test_perf_log = ppo_agent.test_agent(best_rl_agent_path, env, num_episodes=1)
			agent_weights_available.set()  # agent weights are available for deployment thread
			# save the performance data
			rl_perf_save(test_perf_log_list=test_perf_log, log_dir=kwargs['rl_perf_data'],
							 save_as= 'csv', header=writeheader)
			writeheader = False  # don't write header after first iteration

			interval += 1

			print("******End Control Learn Loop 1 iteration*******")

			# if no more learning is needed end this thread
			if to_break:
				break
			if end_learning.is_set():  # break after the next loop
				to_break = True
