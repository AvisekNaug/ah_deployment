## alumni_v2

This repo is used to deploy a reinforcement learning controller on the Alumni Hall. It will run following processes in parallel.

* Actual controller deployment script
* Online Data Collection script
* Offline Data collecting script
* Data-driven model learning script
* Controller learning script
* Off-line controller learning script

## Clone the repo

git clone -b dev_v3 git@github.com:AvisekNaug/ah_deployment.git

## Installation of packages

1. Create virtual environment using (**Optional** but recommended step: This will have a clean and separate installation procedure for python packages that will not mess with existing applications in the server where it will run)
	```bash
	$ python3 -m venv alumni_v2
	$ source alumni_v2/bin/activate
	```
	("python3 -m venv alumni_v2" might generate an error/warning on some Linux systems and it means an additional prerequisite needs to be fulfilled. I don't remember the exact details of the error as it has been a long time but in case it arises please get back to me with the error log and I will try to send out the fix.)

2. Install necessary packages: 

	i. **On Ubuntu** : sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev

	ii. **On Centos** : yum install -y cmake openmpi-devel zlib-devel mesa-libGL-devel

	Also **on Censtos** depending on your python verion you have to do : ```yum install -y python3X-devel``` where is X is 6,7,8 etc. Note the last ```yum search python3*X*-devel``` only provides devel for version .6 nd 3.8. So please make sure you have python with eiter version 3.6 or 3.8


3. Install all requirements

	**For Ubuntu OS**:
	```bash
	$ pip install -r requirements.txt
	```
	or depending on your system,
	```bash
	$ pip3 install -r requirements.txt
	```
	
	**For Debian OS**:
	```bash
	env MPICC=/usr/lib64/openmpi/bin/mpicc pip install -r requirements.txt
	```

4. Exit the environment(**Only** to be done in case step 1 has been followed):
	```bash
	$ deactivate
	```

## Starting the scripts

1. **In case**, virtual environment "alumni_v2" from step 1 has been created, you have to activate it at the location where it was created
	```bash
	$ source alumni_v2/bin/activate
	```

1. Launching online learning script(Mandatory)
	```bash
	$ nohup python main.py -o 0.70 > main.out
	```

2. Start the script which calculates the wet bulb temperature in a seperate shell(Mandatory)
	```bash
	$ nohup python wbt_calculator.py > wbt_calculator.out
	```
Step 4 is **not needed** for Alumni Deployment

4. **In case** you want to run the production facing server to visualize a live Dashboard, open a new terminal and execute the following
	```bash
	$ waitress-serve --host <server ip address> --port <port to run> live_plot:app.server
	```

	You can visualize the live dashboard by going to the IP address of the server followed by the port where you set it up.

<!-- 3. Install jupyter notebook extensions for rich extensions: Navigate to "Extensions" after notebook launch to enable desired extensions like Hinterland, Cell Collapse etc
	```bash
	pip install notebook
	pip install jupyter_contrib_nbextensions
	jupyter contrib nbextension install --user
	pip install jupyter_nbextensions_configurator
	jupyter nbextensions_configurator enable --user
	``` -->