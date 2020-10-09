"""
This script implements the different threads that are going to be executed for the Alumni Hall deployment.
This script will control the "Supply Air Set Point" for the Air Handling Units. The skeleton script 
is now being created to formalize the main components needed in creating the relearning agent.
"""
import os
from os import path
# Enable '0' or disable '' GPU use
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from multiprocessing import Event, Lock
from threading import Thread
from datetime import datetime
import json
import warnings
import time
from datetime import datetime, timedelta
import gc
from argparse import ArgumentParser
import subprocess

# ------------From Ibrahim's controller.py script
import sys
import logging
# Set up logging with a global variable "log"
logging.captureWarnings(True)
log = logging.getLogger(__name__)
_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
_handler = logging.StreamHandler(sys.stderr)
# Initially, before the code in __main__ guard executes, logging is set to DEBUG
# to print out all messages to console.
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(_formatter)
log.addHandler(_handler)
# ------------From Ibrahim's controller.py script

with warnings.catch_warnings():
	from alumni_scripts import deploy_control as dctrl
	from alumni_scripts import alumni_data_utils as a_utils
	from source import utils

parser = ArgumentParser(description='Main script for Alumni Hall RL controlelr')
parser.add_argument('--oat_th', '-o', type=float, default=0.72, help='threshold for oat')
parser.add_argument('--relearn_interval_hours', '-n', type=int, default=168, help='relearning interval')
parser.add_argument('--relearn_weeks', '-w', type=int, default=2, help='weeks to look into the past to train rl agent')
parser.add_argument('--num_of_episodes', '-e', type=int, default=60, help='number of episodes to train')
parser.add_argument('--deploy_interval_mins', '-d', type=int, default=30, help='interval in mins for controller output')
parser.add_argument('--expert_demo', '-x', type=bool, default=True, help='Use expert heuristics to create some initial data')

if __name__ == "__main__":
	
	try:
		args = parser.parse_args()
		# ------------From Ibrahim's controller.py script
		# Specifing the log file name
		_logfile_handler = logging.FileHandler(filename='deploy_log.txt', mode='w')
		_logfile_handler.setLevel(logging.DEBUG)    # DEBUG is the lowest severity. It means print all messages.
		_logfile_handler.setFormatter(_formatter)   # Set up the format of log messages
		log.addHandler(_logfile_handler)            # add this handler to the logger
		# set up logging severity
		log.setLevel(logging.DEBUG)
		# ------------From Ibrahim's controller.py script

		exp_params = {}
		# interval num for relearning : look at logs/Interval{} and write next number to prevent overwrite
		interval = 1
		# how to set relearning interval
		relearn_interval_kwargs = {'days':0, 'hours':args.relearn_interval_hours, 'minutes':0, 'seconds':0}
		# period of data
		period = 6 # 1 = 5 mins, 6 = 30 mins

		save_path = 'tmp/'
		model_path = 'models/'
		log_path = 'logs/'
		results = 'results/'
		trend_data = 'data/trend_data/'
		cwe_data = save_path + 'cwe_data/'
		hwe_data = save_path + 'hwe_data/'
		vlv_data = save_path + 'vlv_data/'
		env_data = save_path + 'env_data/'
		rl_perf_data = save_path + 'rl_perf_data/'

		# path to saved agent weights
		best_rl_agent_path = 'models/best_rl_agent'

		#if not path.exists(cwe_data):
		utils.make_dirs(cwe_data)  # prevent data overwrite from offline exps
		#if not path.exists(hwe_data):
		utils.make_dirs(hwe_data)  # prevent data overwrite from offline exps
		#if not path.exists(vlv_data):
		utils.make_dirs(vlv_data)  # prevent data overwrite from offline exps
		#if not path.exists(env_data):
		utils.make_dirs(env_data)  # prevent data overwrite from offline exps
		if not path.exists(model_path):
			utils.make_dirs(model_path)  # prevent data overwrite from offline exps
		# if not path.exists(log_path):
		utils.make_dirs(log_path)  # prevent data overwrite from offline exps
		#utils.make_dirs(results)
		# if not path.exists(rl_perf_data):
		utils.make_dirs(rl_perf_data)  # prevent data overwrite from offline exps

		exp_params['env_config'] = {
			'save_path' : env_data, 'model_path': model_path, 'logs' : log_path,
			'obs_space_vars' : ['oat', 'oah', 'wbt', 'avg_stpt', 'sat'], 
			'action_space_vars' :['sat'],  # has to be same as sat since we are effectively replacing new sat with sat_stpt
			'cwe_inputs' : ['sat-oat', 'oah', 'wbt', 'pchw_flow'],
			'hwe_inputs' : ['oat', 'oah', 'wbt', 'sat-oat'],
			'vlv_inputs' : ['oat', 'oah', 'wbt', 'sat-oat'],
		}

		# Events
		end_learning = Event()  # end the relearning procedure
		agent_weights_available = Event()  # agent weights are available to be read by deploy loop
		agent_weights_available.set() # set agent weights to available in online learning as it
		# will be immediately deployed

		# Locks
		agent_weights_lock = Lock()  #  read / write rl weights without access issues
		
		# get agg type and data stats from meta_data.json
		with open('alumni_scripts/meta_data.json', 'r') as fp:
			meta_data_ = json.load(fp)
		scaler = a_utils.dataframescaler(meta_data_['column_stats_half_hour'])
		log.info("------------Relearn Thread Started------------------")

		deploy_ctrl_th = Thread(target=dctrl.deploy_control, daemon = False,
							kwargs={'agent_weights_available' : agent_weights_available,
									'agent_weights_lock' : agent_weights_lock,
									'obs_space_vars' : exp_params['env_config']['obs_space_vars'],
									'scaler' : scaler,
									'best_rl_agent_path' : best_rl_agent_path,
									'period' : period,
									'end_learning': end_learning,
									'deploy_interval_mins':args.deploy_interval_mins,
									'expert_demo':args.expert_demo,
									'logger':log,})
		log.info("Main Thread: Deployment Thread Started")
		deploy_ctrl_th.start()


		while not end_learning.is_set():

			try:
				last_relearn_time = datetime.now()
				

				r = subprocess.run(['python', 'online_learning.py', '-i', str(interval), '-o', str(args.oat_th), \
					'-w', str(args.relearn_weeks), '-e', str(args.num_of_episodes)])

				log.info("Main Thread: Relearn Script Terminated: Agent can try new weights")
				agent_weights_available.set()

				log.info('Main Thread: Running Garbage collection')
				gc.collect()

				# increase interval value for storing rewards
				interval += 1

				# sleep for relearn_interval before next relearn stage
				to_sleep = timedelta(**relearn_interval_kwargs)+last_relearn_time-datetime.now()
				log.info('Main Thread: Next Relearn in {} minutes'.format(str(to_sleep)))
				time.sleep(to_sleep.total_seconds())


			except KeyboardInterrupt:
				end_learning.set()
				log.info('Main Thread: Exiting Main Thread')
				
		deploy_ctrl_th.join()
		log.info('Exiting deployment thread')
	
	except Exception as e:
		log.critical('Main Thread: Script stopped due to:\n%s', str(e))
		log.debug(e, exc_info=True)
		exit(-1)
