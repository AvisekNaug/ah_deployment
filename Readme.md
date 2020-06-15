## alumni_v2

This repo is used to deploy a reinforcement learning controller on the Alumni Hall. It will run following processes in parallel.

* Data-driven model learning script
* Off-line controller learning script
* Actual controller deployment script
* Data collecting script

## Installation of packages

1. Create virtual environment using 
	```bash
	python -m venv alumni_v2
	source phm20/bin/activate
	```

2. Install all requirements
	```bash
	pip install -r minimalrequirements.txt
	```

3. Install jupyter notebook extensions for rich extensions: Navigate to "Extensions" after notebook launch to enable desired extensions liek Hinterland, Cell Collapse etc
	```bash
	pip install jupyter_contrib_nbextensions
	jupyter contrib nbextension install --user
	pip install jupyter_nbextensions_configurator
	jupyter nbextensions_configurator enable --user
	```