"""
Custom env from identical to alumnienv but with extra methods to adjust env properties. 
Essential for use with SubprocVecEnv
"""
import os

import numpy as np
import pandas as pd

import gym
from gym import spaces
from gym.utils import seeding


class Env(gym.Env):

	def __init__(self,
				 df,totaldf_stats,
				 obs_space_vars : list,
				 action_space_vars : list,
				 action_space_bounds: list, 
				 cwe_energy_model, cwe_input_shape, cwe_input_vars,
				 hwe_energy_model, hwe_input_shape, hwe_input_vars,
				 vlv_state_model, vlv_input_shape, vlv_input_vars,
				 slicepoint = 0.75,
				 **kwargs):

		self.re_init_env(df,totaldf_stats,
				 obs_space_vars,
				 action_space_vars,
				 action_space_bounds, 
				 cwe_energy_model, cwe_input_shape, cwe_input_vars,
				 hwe_energy_model, hwe_input_shape, hwe_input_vars,
				 vlv_state_model, vlv_input_shape, vlv_input_vars,
				 slicepoint =slicepoint,
				 **kwargs)

		'''Begin: standard requirements for interfacing with gym environments'''
		# collect dataframe statistics in a dataframe
		self.stats = totaldf_stats
		# the action space bounds
		assert len(action_space_bounds)==2, "Exactly two elements: low and high bounds are needed"
		assert all([actlow == -1*acthigh for actlow,
		 acthigh in zip(action_space_bounds[0], action_space_bounds[1])]), "Action Space bounds have to be symmetric"
		self.action_space_bounds = action_space_bounds
		self.action_space = spaces.Box(low = np.array(self.action_space_bounds[0]),
										high = np.array(self.action_space_bounds[1]),
										dtype = np.float32)

		# the observation space (it is assumed to be continuous)
		spacelb = self.stats.loc['min', self.obs_space_vars].to_numpy().flatten()
		spaceub = self.stats.loc['max', self.obs_space_vars].to_numpy().flatten()
		self.observation_space = spaces.Box(low = spacelb, high = spaceub, dtype =  np.float32)

		# other base environment requirements
		# self.seed()
		self.viewer = None
		'''End: standard requirements for interfacing with gym environments'''

	def re_init_env(self, df, totaldf_stats,
				 obs_space_vars : list,
				 action_space_vars : list,
				 action_space_bounds: list, 
				 cwe_energy_model, cwe_input_shape, cwe_input_vars,
				 hwe_energy_model, hwe_input_shape, hwe_input_vars,
				 vlv_state_model, vlv_input_shape, vlv_input_vars,
				 slicepoint = 0.75,
				 **kwargs):
		"""Custom method to reinitialize the environment
		"""
		# the iterable dataframe
		self.df = df
		self.nrows, self.ncols = df.shape

		# make sure observation and action variables are part of the dataframe
		condition1 = all(name in self.df.columns for name in obs_space_vars)
		condition2 = all(name in self.df.columns for name in action_space_vars)
		assert condition1 & condition2 , "Both action and observation space variables must be in the input dataframe"
		# the observation space variable names list
		self.obs_space_vars = obs_space_vars
		# the action space variables names list
		self.action_space_vars = action_space_vars


		'''Begin: Requirements for our own environment'''
		# cwe energy model
		self.cwe_model = cwe_energy_model
		self.cwe_input_shape = cwe_input_shape
		self.cwe_input_vars = cwe_input_vars
		# hwe energy model
		self.hwe_model = hwe_energy_model
		self.hwe_input_shape = hwe_input_shape
		self.hwe_input_vars = hwe_input_vars
		# valve state model
		self.vlv_model = vlv_state_model
		self.vlv_input_shape = vlv_input_shape
		self.vlv_input_vars = vlv_input_vars

		# testing: whether we env is in testing phase or not
		self.testing = False
		# dataptr: Steps through the entire available data in a cycle. 
		# Gets reset to the train data start index when entire train data is used up. 
		# Gets reset to the test data start index when entire test data is used up.
		self.dataptr = 0
		# Steps for specifying train and test data indices
		self.slicepoint = slicepoint
		self.train_data_limit = int(self.slicepoint * self.nrows)
		self.test_data_limit = self.nrows
		# episode_length: dictates number of steps in an episode
		self.episode_length = 1544  # self.train_data_limit 20
		# Used to store the old historical action for comparison
		self.hist_a = None
		# Information about the reward params
		self.params = kwargs
		'''End: Requirements for our own environment'''

	def reset(self,):

		# Helps iterate through the data file. This is not the actual observation
		self.row = self.df.iloc[[self.dataptr],:].copy()
		# Collect part of the observations for next time step
		self.s = self.row.loc[:, self.obs_space_vars]
		# step_counter: counts the current step number in an episode
		self.step_counter = 0
		'''Standard requirements for interfacing with gym environments'''
		self.steps_beyond_done = None

		return self.s.to_numpy().flatten()  # flatten them before feeding to the rl agent

	def step(self, a):

		# Advance datapointer to next row in the dataframe
		if not self.testing:
			self.dataptr = self.dataptr + 1 if self.dataptr < self.train_data_limit - 1 else 0
		else:
			self.dataptr = self.dataptr + 1 if self.dataptr < self.test_data_limit - 1 else self.train_data_limit

		# process control action
		a = self.process_action(self.s, a)

		# step to the next state and get new observation
		self.s_next = self.state_transition(self.s, a)

		# calculate reward and collect information for "info" dinctionary
		r, step_info = self.reward_calculation(self.s, a, self.s_next)
		
		# increment step counter
		self.step_counter += 1
		# check if episode has terminated
		done = False
		if self.step_counter > self.episode_length -1 :
			done = True  # episode has ended

		# proceed to the next state if not finished
		if not done:
			self.s = self.s_next

		return self.s_next.to_numpy().flatten(), float(r), done, step_info


	def state_transition(self, s, a):
		"""
		Custom method for state transition model
		"""

		# Helps iterate through the data file to get info of NEXT time step. This is not the actual observation
		self.row = self.df.iloc[[self.dataptr],:].copy()

		# cache the old action before removing it
		self.cache_hist_action()

		# Update historical action cells with actual agent action values
		self.row.loc[:,self.action_space_vars] = a

		# Collect observations for next time step and also return processed action
		return self.row.loc[:, self.obs_space_vars]


	def reward_calculation(self, s, a, _):
		"""Custom Reward Calculator
		"""

		'''Energy reward'''
		# calculate rl based energy for the this time step
		cwe_rl_energy, hwe_rl_energy = self.energy_cost(s, a)
		# historical energy for this time step
		cwe_hist_energy, hwe_hist_energy = self.energy_cost(s, self.hist_a)
		rl_energy = hwe_rl_energy #+ cwe_rl_energy
		hist_energy = hwe_hist_energy #+  cwe_hist_energy

		# 'energy_saved' reward if at least 'energy_savings_thresh' energy saved else 'energy_penalty' reward
		reward_energy = self.params['energy_saved']*(hist_energy-rl_energy) \
			if hist_energy-rl_energy>self.params['energy_savings_thresh'] \
			else self.params['energy_penalty']*(-hist_energy+rl_energy)
		# reward_energy = hist_energy-rl_energy
		# reward_energy *= self.params['energy_reward_weight']  # don't weight it so that we can try ad hoc weights
		# reward_energy /= self.episode_length  # scale reward in case the episode lengths are not equal

		'''Comfort Reward'''
		# exrtact rl discharge air temeprature
		T_rl_disch = a[0]
		# extract vrf average setpoint temperature
		avg_vrf_stpt = s.loc[s.index[0], 'avg_stpt']
		# 'comfort' reward if T_rl_disch close to avg_vrf_stpt by 'comfort_thresh' else 'uncomfort' reward
		reward_comfort = self.params['comfort']/(abs(T_rl_disch-avg_vrf_stpt) + 0.1) \
			if abs(T_rl_disch-avg_vrf_stpt) < self.params['comfort_thresh'] \
			else self.params['uncomfortable']*abs(T_rl_disch-avg_vrf_stpt)
		# reward_comfort = -1*abs(T_rl_disch-avg_vrf_stpt)
		# reward_comfort *= self.params['comfort_reward_weight']  # don't weight it so that we can try ad hoc weights
		# reward_comfort /= self.episode_length  # scale reward in case the episode lengths are not equal

		# TODO: Create error component

		reward = self.params['energy_reward_weight']*reward_energy + self.params['comfort_reward_weight']*reward_comfort

		if self.testing:
			step_info = {'datetime': str(s.index[0]),
						'rl_cwe': cwe_rl_energy,
						'rl_hwe': hwe_rl_energy,
						'hist_cwe':cwe_hist_energy,
						'hist_hwe':hwe_hist_energy,
						'reward_energy': reward_energy,
						'reward_comfort': reward_comfort,
						'oat': s.loc[s.index[0], 'oat'],
						'orh': s.loc[s.index[0], 'orh'],
						# 'rl_sat': s.loc[s.index[0], 'sat'],
						'T_rl_disch': T_rl_disch,
						'avg_stpt': s.loc[s.index[0], 'avg_stpt']
						}
		else:
			step_info = {}

		return reward, step_info


	def energy_cost(self, s, a):
		"""Custom energy calculator
		"""

		# Since for our case, the action modifies the state sirectly, \
		# we replace action columns with corresponding rl action
		assert all(name in s.columns for name in self.action_space_vars), \
			"Action space vars are not part of observation space vars"
		
		# do not modify original state s
		in_obs = s.copy()
		# change old actions to new actions
		in_obs.loc[:, self.action_space_vars] = a

		"""create inputs for cwe, hwe and vlv_state"""
		# get input to hwe model: First do some mathematical trickery to get inputs outside obs space
		# need additional data from df which are not part of the state/observation space
		temp_row = self.df.iloc[[self.dataptr-1],:].copy()  # the -1 there since dataptr has davanced to nxt state
		# update temp_row columns with in_obs data
		temp_row.loc[:, in_obs.columns] = in_obs.to_numpy().flatten()
		
		# CHANGE THE SAT_OAT difference
		temp_row['sat-oat'] = temp_row['sat']-temp_row['oat']

		"""cololing energy prediction"""
		# get input to cwe model
		cwe_in_obs = temp_row.loc[:, self.cwe_input_vars]  # in_obs.loc[:, self.cwe_input_vars]
		# convert to array and reshape
		cwe_in_obs = cwe_in_obs.to_numpy().reshape(self.cwe_input_shape)
		# calculate cooling energy
		cooling_energy = float(self.cwe_model.predict(cwe_in_obs, batch_size=1).flatten())

		"""valve state prediction"""
		# now select the inputs
		vlv_in_obs = temp_row.loc[:, self.vlv_input_vars]
		# convert to array and reshape
		vlv_in_obs = vlv_in_obs.to_numpy().reshape(self.vlv_input_shape)
		# calculate valve state as 0 = off or 1 = on
		valve_state =  np.argmax(self.vlv_model.predict(vlv_in_obs, batch_size=1).flatten())

		"""heating energy prediction only if valve is on = 1"""
		if valve_state == 1:
			# now select the inputs
			hwe_in_obs = temp_row.loc[:, self.hwe_input_vars]
			# convert to array and reshape
			hwe_in_obs = hwe_in_obs.to_numpy().reshape(self.hwe_input_shape)
			# calculate heating energy
			heating_energy =  float(self.hwe_model.predict(hwe_in_obs, batch_size=1).flatten())
		else:
			heating_energy = 0.0

		return cooling_energy, heating_energy


	def process_action(self, s, a):
		"""Custom action processer which depends on current state
		"""
		# Action variable upper and lower bounds for the dataframe provided
		a_lowerbound = self.stats.loc['min', self.action_space_vars].to_numpy().flatten()
		a_upperbound = self.stats.loc['max', self.action_space_vars].to_numpy().flatten()

		#  Scale action differential from gym space(which is also raw data space) to scaled space
		a = a/(self.params['action_minmax'][1] - self.params['action_minmax'][0])

		# Supply Temperature at previous state
		previous_sat = s.loc[s.index[0], self.action_space_vars].to_numpy().flatten()

		# new action: previous state + new action differential
		a += previous_sat

		# clip action
		a = np.clip(a, a_lowerbound, a_upperbound)

		return a


	def testenv(self,):
		self.testing = True
		self.dataptr = self.train_data_limit


	def trainenv(self,):
		self.testing = False
		self.dataptr = 0

	def cache_hist_action(self,):

		# We look at one dataptr behind updated dataptr since we wwant the 
		# action at state s, not state s_next. dataptr is updated by 1 at the 
		# beginning of the step function. That is why we move it back by 1 unit.
		temp_row = self.df.iloc[[self.dataptr-1],:].copy()
		self.hist_a = temp_row.loc[:,self.action_space_vars].copy().to_numpy().flatten()


