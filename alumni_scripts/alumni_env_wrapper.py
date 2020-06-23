import os
import warnings
import csv
import json
from typing import Optional
import time

import gym


from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

def custom_make_vec_env(env_id, n_envs=1, seed=None, start_index=0,
				 monitor_dir=None, wrapper_class=None,
				 env_kwargs=None, vec_env_cls=None, vec_env_kwargs=None):
	"""
	Create a custom wrapped, monitored `VecEnv`.
	By default it uses a `DummyVecEnv` which is usually faster
	than a `SubprocVecEnv`.

	:param env_id: (str or Type[gym.Env]) the environment ID or the environment class
	:param n_envs: (int) the number of environments you wish to have in parallel
	:param seed: (int) the initial seed for the random number generator
	:param start_index: (int) start rank index
	:param monitor_dir: (str) Path to a folder where the monitor files will be saved.
		If None, no file will be written, however, the env will still be wrapped
		in a Monitor wrapper to provide additional information about training.
	:param wrapper_class: (gym.Wrapper or callable) Additional wrapper to use on the environment.
		This can also be a function with single argument that wraps the environment in many things.
	:param env_kwargs: (dict) Optional keyword argument to pass to the env constructor
	:param vec_env_cls: (Type[VecEnv]) A custom `VecEnv` class constructor. Default: None.
	:param vec_env_kwargs: (dict) Keyword arguments to pass to the `VecEnv` class constructor.
	:return: (VecEnv) The wrapped environment
	"""
	env_kwargs = {} if env_kwargs is None else env_kwargs
	vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs

	def make_env(rank):
		def _init():
			if isinstance(env_id, str):
				env = gym.make(env_id)
				if len(env_kwargs) > 0:
					warnings.warn("No environment class was passed (only an env ID) so `env_kwargs` will be ignored")
			else:
				env = env_id(**env_kwargs)
			if seed is not None:
				env.seed(seed + rank)
				env.action_space.seed(seed + rank)
			# Wrap the env in a Monitor wrapper
			# to have additional training information
			monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
			# Create the monitor folder if needed
			if monitor_path is not None:
				os.makedirs(monitor_dir, exist_ok=True)
			env = customMonitor(env, filename = monitor_path, rank = rank)
			# Optionally, wrap the environment with the provided wrapper
			if wrapper_class is not None:
				env = wrapper_class(env)
			return env
		return _init

	# No custom VecEnv is passed
	if vec_env_cls is None:
		# Default: use a DummyVecEnv
		vec_env_cls = DummyVecEnv

	return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)

class customMonitor(Monitor):

	def __init__(self, env: gym.Env, filename: Optional[str], rank):
		super(customMonitor, self).__init__(env=env, filename=filename)

		self.myEnv_id = env.spec and env.spec.id  # prevent name clash
		self.myEnv_rank = rank

	def changelogpath(self, filepath: Optional[str], reset_keywords=(), info_keywords=()):
		
		self.t_start = time.time()  # specify new time
		if filepath is None:
			self.file_handler = None
			self.logger = None
		else:
			if self.file_handler is not None:  # close previous file_handler
				self.file_handler.close()
			os.makedirs(filepath, exist_ok=True)
			filepath = os.path.join(filepath, str(self.myEnv_rank))  # add env_id to end of filepath
			if not filepath.endswith(Monitor.EXT):
				if os.path.isdir(filepath):
					filepath = os.path.join(filepath, Monitor.EXT)
				else:
					filepath = filepath + "." + Monitor.EXT
			self.file_handler = open(filepath, "wt")
			self.file_handler.write('#%s\n' % json.dumps({"t_start": self.t_start, 'env_id': self.myEnv_id}))
			self.logger = csv.DictWriter(self.file_handler,
										fieldnames=('r', 'l', 't') + reset_keywords + info_keywords)
			self.logger.writeheader()
			self.file_handler.flush()