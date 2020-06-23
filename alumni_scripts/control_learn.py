"""
This scripts contains methods that run learn the rl agent for control
"""

import warnings
with warnings.catch_warnings():
	from stable_baselines.common.vec_env import DummyVecEnv
	from stable_baselines.common import set_global_seeds, make_vec_env
	from keras.models import load_model
	import alumni_env
	import ppo_agent
	import alumni_env_wrapper

from pandas import read_pickle
from multiprocessing import Event, Lock


def controller_learn(*args, **kwargs):
	"""
	Learn the controller on a data-driven simulator of the environment
	"""
	env_data_available : Event = kwargs['env_data_available']  # new data available for env relearning
	lstm_weights_available : Event = kwargs['lstm_weights_available']  # lstm weights are available for alumni env
	agent_weights_available : Event = kwargs['agent_weights_available']  # deploy loop can read the agent weights now

	env_train_data_lock : Lock = kwargs['env_train_data_lock']  # prevent data read/write access
	lstm_weights_lock : Lock = kwargs['lstm_weights_lock']  # prevent data read/write access
	agent_weights_lock : Lock = kwargs['agent_weights_lock']  # prevent data read/write access

	interval = 1
	env_created = False

	while True:

		# if data for alumni env and energy models are available, then run controller training
		if (env_data_available.is_set() & lstm_weights_available.is_set()):  

			with env_train_data_lock:
				df_scaled = read_pickle(kwargs['save_path']+'env_data.pkl')
			df_scaled_stats = df_scaled.describe()
			env_data_available.clear()

			with lstm_weights_lock:
				cwe_energy_model  =load_model(kwargs['save_path']+'cwe best_model')
				hwe_energy_model  =load_model(kwargs['save_path']+'hwe best_model')
				vlv_energy_model  =load_model(kwargs['save_path']+'vlv best_model')
			lstm_weights_available.clear()

			"""create environment with new data"""
			monitor_dir = kwargs['logs']+'Interval_{}/'.format(interval)

			
			"""Arguments to be fed to the custom environment inside make_vec_env"""
			reward_params = {'energy_saved': 100.0, 'energy_savings_thresh': 0.0,
							 'energy_penalty': -100.0, 'energy_reward_weight': 0.5,
							 'comfort': 1.0, 'comfort_thresh': 0.10,
							 'uncomfortable': -1.0, 'comfort_reward_weight': 0.5,}
			env_kwargs = dict(  #  Optional keyword argument to pass to the env constructor
				df = df_scaled,
				totaldf_stats = df_scaled_stats,
				obs_space_vars=kwargs['obs_space_vars'],
				action_space_vars=kwargs['action_space_vars'],
				action_space_bounds=[[-2.0], [2.0]],  # bounds for real world action space; is scaled
				# internally using the reward_params

				cwe_energy_model=cwe_energy_model,
				cwe_input_vars=kwargs['cwe_inputs'],
				cwe_input_shape=(1, 1, len(kwargs['cwe_inputs'])),

				hwe_energy_model=hwe_energy_model,
				hwe_input_vars=kwargs['hwe_inputs'],
				hwe_input_shape=(1, 1, len(kwargs['hwe_inputs'])),

				vlv_state_model=vlv_energy_model,
				vlv_input_vars=kwargs['vlv_inputs'],
				vlv_input_shape=(1, 1, len(kwargs['vlv_inputs'])),

				**reward_params  # the reward adjustment parameters
			)

			env_id = alumni_env.Env  # the environment ID or the environment class
			start_index = 0  # start rank index
			vec_env_cls = DummyVecEnv

			if not env_created:
				env = alumni_env_wrapper.custom_make_vec_env(
					env_id=env_id, n_envs=1, start_index = start_index, monitor_dir = monitor_dir,
					vec_env_cls = vec_env_cls, env_kwargs = env_kwargs
				)
				env_created = True
			else:	
				# change the monitor log directory
				env.env_method('changelogpath', (monitor_dir))
				# reinitialize the environment with new data and models
				env.env_method('re_init_env', **env_kwargs)

			"""provide envrionment to the new or existing rl model"""
			




	