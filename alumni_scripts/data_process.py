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

	# tz_utc = timezone(offset=-timedelta(hours=0))
	tz_central = timezone(offset=-timedelta(hours=6))

	start = datetime(kwargs['start_year'], kwargs['start_month'], kwargs['start_day'], hour= kwargs['start_hour'],
	 minute= kwargs['start_minute'], second= kwargs['start_second'], tzinfo=tz_central)
	end   = datetime(kwargs['end_year'], kwargs['end_month'], kwargs['end_day'], hour= kwargs['end_hour'],
	 minute= kwargs['end_minute'], second= kwargs['end_second'], tzinfo=tz_central)

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
	with open(kwargs['meta_data_path'], 'r') as fp:
		meta_data = json.load(fp)

	# Perform 2 standard deviation based thresholding


# online data clean
def online_data_clean(*args, **kwargs):

	pass

