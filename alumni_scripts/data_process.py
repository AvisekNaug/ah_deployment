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

