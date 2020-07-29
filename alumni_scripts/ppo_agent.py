"""
implements a ppo agent for the Alumni Hall
"""
import os
import numpy as np
import glob
import re

import csv
import json
import pandas as pd

import warnings
with warnings.catch_warnings():
	import tensorflow as tf
	from stable_baselines import PPO2
	from stable_baselines.results_plotter import ts2xy

# current best mean reward
best_mean_reward = -np.inf
#steps completed
total_time_steps = 0


def get_agent(env,
			 model_save_dir = '../models/controller/', 
			 monitor_log_dir = '../log/Trial_0/',
			 logger = None):
	"""
	The Proximal Policy Optimization algorithm combines ideas from A2C
	(having multiple workers) and TRPO (it uses a trust region to improve the actor)
	"""

	# Custom MLP policy
	policy_net_kwargs = dict(act_fun=tf.nn.tanh,
						net_arch=[dict(pi=[64, 64], 
						vf=[64, 64])])

	# Create the agent
	agent = PPO2("MlpPolicy", 
				env, 
				policy_kwargs=policy_net_kwargs, 
				verbose=0)

	agent.is_tb_set = False  # attribute for callback
	agent.model_save_dir = model_save_dir  # load or save model here
	agent.monitor_log_dir = monitor_log_dir  # logging directory for current Trial
	agent.rl_logger = logger

	return agent

def train_agent(agent, env=None, steps=30000, tb_log_name = "../log/ppo2_event_folder"):
	"""
	Train the agent on the environment
	"""

	global best_mean_reward
	best_mean_reward = -np.inf

	if env is not None:
		agent.set_env(env)
	
	trained_agent = agent.learn(total_timesteps=steps, callback=CustomCallBack)

	return trained_agent

def CustomCallBack(_locals, _globals):
	"""
	Store neural network weights during training if the current episode's
	performance is better than the previous best performance.
	"""
	self_ = _locals['self']
	log = self_.rl_logger
	global best_mean_reward, total_time_steps

	if not self_.is_tb_set:
		# Do some initial logging setup

		"""Do some stuff here for setting up logging: eg tb_log the weights"""

		# reverse the key
		self_is_tb_set = True  # pylint: disable=unused-variable

	# Print stats every 1000*self_.env.num_envs calls, since for PPO it is called at every n_step
	if (total_time_steps) % (1000*self_.env.num_envs) == 0:
		# Evaluate policy training performance
		# if np.any(_locals['masks']):  # if the current update step contains episode termination
			# prepare csv files to look into
		x_list = []
		y_list = []
		for env_id in range(self_.env.num_envs):
			monitor_files = glob.glob(self_.monitor_log_dir+'**/'+str(env_id)+'.monitor.csv')
			# new method to sort goes  here
			interval_num = []
			for fname in monitor_files:
				interval_num.append(int(re.split(r'(\d+)', fname)[1]))
			interval_num_sorted = sorted(interval_num)
			sorted_monitor_files = []
			sorted_monitor_files = ['{}Interval_{}/{}.monitor.csv'.format(self_.monitor_log_dir,i,env_id) for i in interval_num_sorted]

			x, y = ts2xy(custom_load_results([sorted_monitor_files[-1]]), 'episodes')
			x_list.append(x)
			y_list.append(y)

		# all environments have at least one episode data row in monitor.csv
		if all([len(x) > 0 for x in x_list]):  
			# Average reward for last 5 episodes across all environments in one go by using None
			mean_reward = np.mean(np.array(y_list)[:,-5:], axis = None)
			# Average across all environments in one go by using None
			log.info('Control Learn Module: An average of {} episodes completed'.format(np.mean(np.array(x_list)[:,-1], axis = None)))
			# Compare Reward
			log.info("Control Learn Module: Best mean reward: {:.2f} - Latest 5 sample mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
			# New best model, you could save the agent here
			if mean_reward > best_mean_reward:
				best_mean_reward = mean_reward
				# Example for saving best model
				log.info("Control Learn Module:: Saving New Best Model")
				self_.save(self_.model_save_dir + 'best_rl_agent')
	total_time_steps += self_.env.num_envs

	return True

def test_agent(agent_weight_path: str, env, num_episodes = 1):

	# load agent weights
	agent = PPO2.load(agent_weight_path, 
					env)
	
	"""
	num_envs is an attribute of any vectorized environment
	NB. Cannot use env.get_attr("num_envs") as get_attr return attributes of the base 
	environment which is being vectorized, not attributes of the VecEnv class itself.
	
	"""
	# create the list of classes which will help store the performance metrics
	perf_metrics_list = [performancemetrics() for _ in range(env.num_envs)]

	for _ in range(num_episodes):

		# issue episode begin command for all the environments
		for perf_metrics in perf_metrics_list: perf_metrics.on_episode_begin()
		# reset all the environments
		obslist = env.reset()
		# done_track contains information on which envs have completed the episode
		dones_trace = [False]*env.num_envs
		# set all_done to be true only when all envs have completed an episode
		all_done = all(dones_trace)

		# Step through the environment till all envs have finished current episode
		while not all_done:
			action, _ = agent.predict(obslist)
			obslist, _, doneslist, infotuple = env.step(action)

			# update dones_trace with new episode end information
			dones_trace = [i | j for i,j in zip(dones_trace,doneslist)]

			# update all_done
			all_done = all(dones_trace)

			for idx, done in enumerate(dones_trace):
				if not done:  # log info from only those environments which have not terminated
					perf_metrics_list[idx].on_step_end(infotuple[idx])  # log the info dictionary
			
		# end episode command issued for all environments
		for perf_metrics in perf_metrics_list: perf_metrics.on_episode_end()

	return perf_metrics_list

class performancemetrics():
	"""
	Store the history of performance metrics. Useful for evaluating the
	agent's performance:
	"""

	def __init__(self):
		self.metriclist = []  # store multiple performance metrics for multiple episodes
		self.metric = {}  # store performance metric for each episode

	def on_episode_begin(self):
		self.metric = {}  # flush existing metric data from previous episode

	def on_episode_end(self):
		self.metriclist.append(self.metric)

	def on_step_end(self, info = {}):
		for key, value in info.items():
			if key in self.metric:
				self.metric[key].append(value)
			else:
				self.metric[key] = [value]

def custom_load_results(monitor_files : list):
	"""
	Load all Monitor logs for a given environment across multiple intervals

	:param path: (list) the list of relative log file paths
	:return: (Pandas DataFrame) the logged data
	"""
	# Changes: remove path and introduce relative file path lists
	# Changes: do not sort on time as new environmnet is not time sensitive
	data_frames = []
	headers = []
	for file_name in monitor_files:
		with open(file_name, 'rt') as file_handler:
			if file_name.endswith('csv'):
				first_line = file_handler.readline()
				assert first_line[0] == '#'
				header = json.loads(first_line[1:])
				data_frame = pd.read_csv(file_handler, index_col=None)
				headers.append(header)
			elif file_name.endswith('json'):  # Deprecated json format
				episodes = []
				lines = file_handler.readlines()
				header = json.loads(lines[0])
				headers.append(header)
				for line in lines[1:]:
					episode = json.loads(line)
					episodes.append(episode)
				data_frame = pd.DataFrame(episodes)
			else:
				assert 0, 'unreachable'
			data_frame['t'] += header['t_start']
		data_frames.append(data_frame)
	data_frame = pd.concat(data_frames)
	# data_frame.sort_values('t', inplace=True)
	data_frame.reset_index(inplace=True)
	data_frame['t'] -= min(header['t_start'] for header in headers)
	# data_frame.headers = headers  # HACK to preserve backwards compatibility
	return data_frame
