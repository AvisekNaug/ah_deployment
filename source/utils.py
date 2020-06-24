"""
Utilities to perform some common functions
"""
import os
import shutil

def make_dirs(dir_path):
	try:
		os.makedirs(dir_path)
	except FileExistsError:
		files = os.listdir(dir_path)
		for f in files:
			try:
				shutil.rmtree(dir_path + f)
			except NotADirectoryError:
				os.remove(dir_path + f)