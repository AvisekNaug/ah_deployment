import multiprocessing
import psutil
import numpy as np
from CoolProp.HumidAirProp import HAPropsSI
from os import path, remove
from pandas import read_csv, to_datetime
from dateutil import tz
from alumni_scripts import alumni_data_utils as a_utils
import time
from datetime import timedelta

def calculate_wbt(all_args):
	t_db, rh, cpu_id = all_args
	proc = psutil.Process()
	proc.cpu_affinity([cpu_id])
	T = HAPropsSI('T_wb','R',rh,'T',t_db,'P',101325)
	return T

# convert 
while True:
	if path.exists('data/trend_data/alumni_data_train.csv'):
		time.sleep(timedelta(seconds=40).seconds) # give some time to finish writing
		df_ = read_csv('data/trend_data/alumni_data_train.csv', )
		remove('data/trend_data/alumni_data_train.csv')
		df_ = a_utils.dropNaNrows(df_)

		rh = df_['WeatherDataProfile humidity']/100
		rh = rh.to_numpy()
		t_db = 5*(df_['AHU_1 outdoorAirTemp']-32)/9 + 273.15
		t_db = t_db.to_numpy()

		tdb_rh = np.concatenate((t_db.reshape(-1,1), rh.reshape(-1,1)), axis=1)
		chunks = [ (sub_arr[:, 0].flatten(), sub_arr[:, 1].flatten(), cpu_id)
					for cpu_id, sub_arr in enumerate(np.array_split(tdb_rh, multiprocessing.cpu_count(), axis=0))]
		pool = multiprocessing.Pool()
		individual_results = pool.map(calculate_wbt, chunks)
		# Freeing the workers:
		pool.close()
		pool.join()
		T = np.concatenate(individual_results)

		t_f = 9*(T-273.15)/5 + 32
		df_['wbt'] = t_f

		df_.to_csv('data/trend_data/alumni_data_train_wbt.csv', index=False)

	time.sleep(timedelta(minutes=1).seconds)