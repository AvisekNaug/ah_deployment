"""
This script will have methods/Threads that help create data for different other threads
1. pull_data: This method will acquire data using BdX API

2. offline_data_process: This method is used to create data for off-line learning. It will 
execute the following step in order
a. Get the raw data from multiple sources
b. Clean the data, process(aggregation, smoothing etc) it
c. Store the data in a suitable format  # maintain time stamp information - TSDB

3. data_stats: Create statistics for the data to be used for processing data online
"""

# imports
import numpy as np
import pandas as pd

from bdx import get_trend
from datetime import datetime, timezone, timedelta


def pull_data(*args, **kwargs):

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