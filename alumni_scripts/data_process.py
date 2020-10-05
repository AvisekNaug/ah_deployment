"""
This script will have methods/Threads that help create data for different other threads
1. pull_offline_data: This method will acquire data using BdX API

2. offline_data_clean: This method is used to create data for off-line learning. It will 
execute the following step in order
a. Get the raw data from multiple sources eg csv files
b. Clean the data
c. Store the data in a suitable format  # maintain time stamp information - TSDB

"""

# imports
import numpy as np
import pandas as pd
import json

from bdx import get_trend
from datetime import datetime, timezone, timedelta



# acquire data using BdX API
def pull_offline_data(*args, **kwargs):
	"""
	This method pulls data from the Niagara Networks using the BdX API
	"""

	tz_utc = timezone(offset=-timedelta(hours=0))
	# tz_central = timezone(offset=-timedelta(hours=6))

	start = datetime(kwargs['start_year'], kwargs['start_month'], kwargs['start_day'], hour= kwargs['start_hour'],
	 minute= kwargs['start_minute'], second= kwargs['start_second'], tzinfo=tz_utc)
	end   = datetime(kwargs['end_year'], kwargs['end_month'], kwargs['end_day'], hour= kwargs['end_hour'],
	 minute= kwargs['end_minute'], second= kwargs['end_second'], tzinfo=tz_utc)

	dataframe = get_trend(trend_id=kwargs['trend_id'],
						  username=kwargs['username'],
						  password=kwargs['password'],
						  start=start,
						  end=end)

	dataframe.to_csv(kwargs['save_path'])


# get data stats from raw data: 
def offline_batch_data_clean(*args, **kwargs):
	"""
	remove outliers from offline batch data
	"""
	# with open(kwargs['meta_data_path'], 'r') as fp:
	# 	meta_data_ = json.load(fp)
	meta_data_ = kwargs['meta_data_']
	meta_data = meta_data_.copy()
	for key, value in meta_data_['column_stats'].items():
		if value['std'] == 0:
			meta_data['column_stats'][key]['std'] = 0.0001  # add small std for constant values
	stats = pd.DataFrame(meta_data['column_stats'])

	# Perform 2 standard deviation based thresholding
	df = kwargs['df'].copy()
	cols = df.columns

	retain = True
	if retain:
		# for col_name in df.columns:
		# 	#df.loc[faulty_idx[col_name], col_name] = stats.loc['mean', col_name]  # set to mean
		# 	t = (np.sign(df - stats.loc['mean', :]).loc[faulty_idx[col_name], col_name] *
		# 	 2 *stats.loc['std', col_name]) + stats.loc['mean', col_name]
		# 	df.loc[faulty_idx[col_name], col_name] = t  # set to upper or lower 2*std bound
		mean = stats.loc['mean',cols].to_numpy()
		std = stats.loc['std',cols].to_numpy()
		lb = mean - 2*std
		ub = mean + 2*std
		df.clip(lower=lb, upper=ub, axis=1, inplace=True)
											
	else:
		faulty_idx = np.abs(df-stats.loc['mean',cols]) >= (2*stats.loc['std',cols])
		df = df[~faulty_idx]  # keep rows which are within bounds

	return df


# get online date
def pull_online_data(*args, **kwargs):
	"""
	This method pulls data from the Niagara Networks using the BdX API
	"""

	tz_utc = timezone(offset=-timedelta(hours=0))
	# tz_central = timezone(offset=-timedelta(hours=6))

	start = datetime(kwargs['start_year'], kwargs['start_month'], kwargs['start_day'], hour= kwargs['start_hour'],
	 minute= kwargs['start_minute'], second= kwargs['start_second'], tzinfo=tz_utc)
	end   = datetime(kwargs['end_year'], kwargs['end_month'], kwargs['end_day'], hour= kwargs['end_hour'],
	 minute= kwargs['end_minute'], second= kwargs['end_second'], tzinfo=tz_utc)

	dataframe = get_trend(trend_id=kwargs['trend_id'],
						  username=kwargs['username'],
						  password=kwargs['password'],
						  start=start,
						  end=end)

	dataframe.to_csv(kwargs['save_path'])


# online data clean
def online_data_clean(*args, **kwargs):

	# with open(kwargs['meta_data_path'], 'r') as fp:
	# 	meta_data_ = json.load(fp)
	meta_data_ = kwargs['meta_data_']
	meta_data = meta_data_.copy()
	for key, value in meta_data_['column_stats'].items():
		if value['std'] == 0:
			meta_data['column_stats'][key]['std'] = 0.0001  # add small std for constant values
	stats = pd.DataFrame(meta_data['column_stats'])

	# Perform 2 standard deviation based thresholding
	df = kwargs['df'].copy()
	cols = df.columns

	retain = True
	if retain:
		# for col_name in df.columns:
		# 	#df.loc[faulty_idx[col_name], col_name] = stats.loc['mean', col_name]  # set to mean
		# 	t = (np.sign(df - stats.loc['mean', :]).loc[faulty_idx[col_name], col_name] *
		# 	 2 *stats.loc['std', col_name]) + stats.loc['mean', col_name]
		# 	df.loc[faulty_idx[col_name], col_name] = t  # set to upper or lower 2*std bound
		mean = stats.loc['mean',cols].to_numpy()
		std = stats.loc['std',cols].to_numpy()
		lb = mean - 2*std
		ub = mean + 2*std
		df.clip(lower=lb, upper=ub, axis=1, inplace=True)
											
	else:
		faulty_idx = np.abs(df-stats.loc['mean',cols]) >= (2*stats.loc['std',cols])
		df = df[~faulty_idx]  # keep rows which are within bounds

	return df

"""
A time based schedule for optimal control testing of Alumni Hall. It wil help the rl generate some good experiences at initial stages of relearning. Experimental. Stopped after first few rounds of relearning.
"""
def initial_learning(time_of_day, oat, prev_setting):
	"""
	Cases:
	* oat>75(24C)
		- 0<time.hour=<4 -> 65.5+np.random.normal(0.8,0.40)
		- 4<time.hour=<9 -> 65.0+np.random.normal(0.8,0.40)
		- 9<time.hour=<18 -> 65.0+np.random.normal(0.8,0.40)
		- 18<time.hour=<21 -> 65.5+np.random.normal(0.1,0.001)
		- 21<time.hour=<24 -> 65.5+np.random.normal(0.8,0.40)
	* 68<oat=<75(20C,24C)
		- 0<time.hour=<4 -> 65.5+np.random.normal(1.0,0.40)
		- 4<time.hour=<9 -> 66.5+np.random.normal(0.8,0.40)
		- 9<time.hour=<18 -> 65.0+np.random.normal(0.8,0.40)
		- 18<time.hour=<21 -> 66.5+np.random.normal(0.1,0.001)
		- 21<time.hour=<24 -> 65.5+np.random.normal(1.0,0.40)
	* 50<oat=<68(10C,20C)
		- 0<time.hour=<4 -> 67.5+np.random.normal(1.0,0.20)
		- 4<time.hour=<9 -> 69.0+np.random.normal(0.8,0.20)
		- 9<time.hour=<18 -> 68.0+np.random.normal(2.0,0.40)
		- 18<time.hour=<21 -> 68.5+np.random.normal(1.8,0.40)
		- 21<time.hour=<24 -> 67.5+np.random.normal(1.0,0.20)
	* oat=<50(10C)
		- 0<time.hour=<4 -> 68.5+np.random.normal(1.0,0.20)
		- 6<time.hour=<9 -> 70.0+np.random.normal(0.8,0.20)
		- 9<time.hour=<18 -> 69.0+np.random.normal(2.0,0.20)
		- 18<time.hour=<21 -> 68.5+np.random.normal(1.8,0.40)
		- 21<time.hour=<24 -> 68.5+np.random.normal(1.0,0.20)
	"""
	prop_stp = 68.0 + np.random.normal(2.0,0.80)
	if oat>75.0:  # 24C
		if (time_of_day.hour<=4) & (time_of_day.hour > 0):
			prop_stp = 65.5 + np.random.normal(0.8,0.40)
		if (time_of_day.hour<=9) & (time_of_day.hour > 4):
			prop_stp = 65.0 + np.random.normal(0.8,0.40)
		if (time_of_day.hour<=18) & (time_of_day.hour > 9):
			prop_stp = 65.0 + np.random.normal(0.8,0.40)
		if (time_of_day.hour<=21) & (time_of_day.hour > 18): 
			prop_stp = 65.5 + np.random.normal(0.1,0.001)
		if (time_of_day.hour<=24) & (time_of_day.hour > 21):
			prop_stp = 65.5 + np.random.normal(0.8,0.40)
	if (oat<=75.0) & (oat > 68.0):  # (20C,24C)
		if (time_of_day.hour<=4) & (time_of_day.hour > 0):
			prop_stp = 65.5+np.random.normal(1.0,0.40)
		if (time_of_day.hour<=9) & (time_of_day.hour > 4):
			prop_stp = 66.5+np.random.normal(0.8,0.40)
		if (time_of_day.hour<=18) & (time_of_day.hour > 9):
			prop_stp = 65.0+np.random.normal(0.8,0.40)
		if (time_of_day.hour<=21) & (time_of_day.hour > 18): 
			prop_stp = 66.5+np.random.normal(0.1,0.001)
		if (time_of_day.hour<=24) & (time_of_day.hour > 21):
			prop_stp = 65.5+np.random.normal(1.0,0.40)
	if (oat<=68.0) & (oat > 50.0):  # (10C,20C)
		if (time_of_day.hour<=4) & (time_of_day.hour > 0):
			prop_stp = 67.5+np.random.normal(1.0,0.20)
		if (time_of_day.hour<=9) & (time_of_day.hour > 4):
			prop_stp = 69.0+np.random.normal(0.8,0.20)
		if (time_of_day.hour<=18) & (time_of_day.hour > 9):
			prop_stp = 68.5+np.random.normal(2.0,0.40)
		if (time_of_day.hour<=21) & (time_of_day.hour > 18): 
			prop_stp = 68.5+np.random.normal(1.8,0.40)
		if (time_of_day.hour<=24) & (time_of_day.hour > 21):
			prop_stp = 67.5+np.random.normal(1.0,0.20)
	if (oat<=50.0) :  # (10C)
		if (time_of_day.hour<=4) & (time_of_day.hour > 0):
			prop_stp = 68.5+np.random.normal(1.0,0.20)
		if (time_of_day.hour<=9) & (time_of_day.hour > 4):
			prop_stp = 70.5+np.random.normal(0.8,0.20)
		if (time_of_day.hour<=18) & (time_of_day.hour > 9):
			prop_stp = 69.0+np.random.normal(2.0,0.20)
		if (time_of_day.hour<=21) & (time_of_day.hour > 18): 
			prop_stp = 68.5+np.random.normal(1.8,0.40)
		if (time_of_day.hour<=24) & (time_of_day.hour > 21):
			prop_stp = 68.5+np.random.normal(1.0,0.20)

	delta_change = np.clip(np.array([prop_stp - prev_setting]),a_min=np.array([-2.0]),a_max=np.array([2.0]))
	return delta_change