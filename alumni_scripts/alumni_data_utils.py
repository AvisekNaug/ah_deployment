"""
This script contains utility functions for alumni hall related processing
"""

"""
This script contains all the data processing activities that are needed
before it can be provided to any other object
"""


import os
import glob
from typing import Union


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


import pandas as pd
import swifter
from scipy import stats
from scipy.fftpack import fft
import scipy.signal as signal


from matplotlib import pyplot as plt
from matplotlib.dates import date2num


# sources of data for which processing can be done
DATASOURCE = ['BdX']
DATATYPE = ['.csv', '.xlsx', '.pkl']


# methods plugin dictionary
PLUGINS = dict()
def register(func):
	"""Register a function as a plug-in to call it via string
	
	Arguments:
		func {python function object} -- function to register
	
	Returns:
		python function object -- return same function
	"""
	global PLUGINS
	PLUGINS[func.__name__] = func

	return func


class readfolder():


	"""
	*List or read multiple files from a Directory or Folder ONLY
	*To read the files use file2df method after initializing
	*To return merged dataframe use bothe file2df and then mergerows
	**If specifying full file path use "readfile" class
	Reads data of same variables from raw sources of types mentioned in DATATYPE
	The individual files should have the same last(-1) dimensions. Note that, it
	will assume that there is a timeseries data column in the files
	"""
	def __init__(self, datadir, fileformat=None, timeformat = '%Y-%m-%d %H:%M:%S', 
	dateheading='Date'):
		"""Lists the files to read from the folder
		
		Arguments:
			datadir {str} -- Folder path to read from
		
		Keyword Arguments:
			fileformat {str} -- Provide the regex in case a selected number of files
			 with a certain pattern should be read. The expression to perform Unix 
			 style pathname pattern expansion. (default: {None})
			timeformat {str} -- Format for datetime parsing
			 (default: {'%Y-%m-%d %H:%M:%S'})
			dateheading {str} -- DateColumn Name in the raw data to read date from
			 (default: {'Date'})
		"""

		#---------ensure files to read are available and in required format------

		# make sure it is an existing directory
		assert os.path.isdir(datadir), "Directory not found"

		# attach forward slash if datadir does not have one
		if not datadir.endswith('/'):
			datadir += '/'

		# list the files in the directory
		if fileformat is None:  # read all files from the directory
			flist = glob.glob(os.path.join(datadir, '*'))
		else:
			flist = glob.glob(os.path.join(datadir, fileformat))

		# check whether list has any of the DATATYPEs that we can read from
		dtype_error = "None of the files in folder " + datadir + " is either of the data types "+\
			 " ".join(DATATYPE)
		# TODO: Remove this line below as it might be redundant
		assert any([fname.endswith(tuple(DATATYPE)) for fname in flist]), dtype_error

		# select files with the required extension
		readlist = [fname for fname in flist if fname.endswith(tuple(DATATYPE))]
		# check whether the is list has one element
		assert readlist, dtype_error
		
		#------------------------------------------------------------------------

		# assign path to read data from
		self.read_list = readlist

		# select time-format
		self.timeformat = timeformat
		self.dateheading = dateheading

		# read empty dataframes into lists
		self.dflist = []

		# create empty datframe variable
		self.df = None


	def return_df(self, processmethods: list = ['files2dflist',\
		 'datetime_parse_dflist']):

		"""Perform all the basic processes on the raw data and return processed
		dataframe in case user is not interested in doing all operations manually
		
		order of ops is persistent in the list: https://stackoverflow.com/questions\
			/13694034/is-a-python-list-guaranteed-to-have-its-elements-stay-\
				in-the-order-they-are-inse
		
		Keyword Arguments:
			processmethods {list} -- List of ops to do
		"""
		opcomplete = None
		for ops in processmethods:
			opcomplete = PLUGINS[ops](self,opcomplete)

		return opcomplete
		

	@register
	def files2dflist(self, *args):
		"""Just reads the raw dataframe in to a list w/o processing it
			This method stores the frames in a self.dflist for future computation
		Returns:
			list -- list of raw dataframes
		"""
		# Read the list of files of same variables from different file types
		for fname in self.read_list:
			self.dflist.append(PLUGINS['read'+
			 os.path.splitext(fname)[1][1:]](fname))

		return self.dflist


	@register
	def datetime_parse_dflist(self, dflist):
		"""Parse the list of dataframes for datetime and set it as index
		
		Returns:
			[type] -- [description]
		"""
		return [datetime_parse(dfiter, self.timeformat, self.dateheading) \
			for dfiter in dflist]


	@register
	def merge_dflist(self, dflist):
		return mergerows(dflist)


class readfile():


	"""
	*List or read a single file
	*To read the files use file2df method after initializing
	*To return a dataframe use bothe file2df and then mergerows
	**If specifying full file path use "readfile" class
	
	Returns:
		[type] -- [description]
	"""

	def __init__(self, filepath, timeformat='%Y-%m-%d %H:%M:%S', dateheading='Date'):

		#---------ensure file to read is available and in required format------

		# make sure it is an existing file
		assert os.path.isfile(filepath), "File not found or specified string is not a file"

		# check whether the file is of one of the readable types
		dtype_error = "File " + os.path.basename(filepath) + " is not of types: "+" ".join(DATATYPE)
		assert filepath.endswith(tuple(DATATYPE)), dtype_error
		
		#------------------------------------------------------------------------

		# assign path to read data from
		self.read_path = filepath

		# select time-format
		self.timeformat = timeformat
		self.dateheading = dateheading

		# create empty dataframe variable
		self.df = None


	def return_df(self, processmethods: list = ['file2df',\
		 'datetime_parse_df']):

		"""Perform all the basic processes on the raw data and return processed
		dataframe in case uesr is not interested in doing all operations manually
		
		order of ops is persistent in the list: https://stackoverflow.com/questions\
			/13694034/is-a-python-list-guaranteed-to-have-its-elements-stay-\
				in-the-order-they-are-inse
		
		Keyword Arguments:
			processmethods {list} -- List of ops to do
		"""
		opcomplete = None
		for ops in processmethods:
			opcomplete = PLUGINS[ops](self, opcomplete)

		return opcomplete


	@register
	def file2df(self, *args):
		"""Reads the raw dataframe w/o procesing it.
		
		Returns:
			[pd.DataFrame] -- raw Dataframe
		"""
		# Read the file
		self.df = PLUGINS['read'+ os.path.splitext(self.read_path)[1][1:]](self.read_path)

		return self.df


	@register
	def datetime_parse_df(self, df):
		return datetime_parse(df, self.timeformat, self.dateheading)


# Static Methods for reading data
@register
def readcsv(read_path):

	df = pd.read_csv(read_path)

	return df


# Static Methods for reading data
@register
def readxlsx(read_path):
	
	df = pd.read_excel(read_path)

	return df


# Static Methods for reading data
@register
def readpkl(read_path):

	df = pd.read_pickle(read_path)

	return df


# Method for parsing data
def datetime_parse(df, timeformat, dateheading):
	"""Convert datetime column to index after parsing it
	
	Arguments:
		df {pd.DataFrame} -- dataframe to parse
		timeformat {str} -- Format for datetime parsing
		dateheading {str} -- DateColumn Name in the raw data to read date from

	Returns:
		[pd.DataFrame] -- parsed dataframe
	"""

	# prevent modifying original dataframe
	dfc = df.copy()

	# Parsing the Date column
	# TODO: parse with timezone information
	dfc.insert(loc=0, column='Time', value=pd.to_datetime(dfc[dateheading], \
		format=timeformat)) # + pd.DateOffset(hours=offset))

	# Drop the original "dateheading" column
	dfc = dfc.drop(dateheading, axis=1)

	# Set Time column as index
	dfc = dfc.set_index(['Time'], drop=True)

	# Dropping duplicated time points that may exist in the data
	dfc = dfc[~dfc.index.duplicated()]

	return dfc


# Methods for cleaning data
def mergerows(dflist):
		"""Merge rows of dataframes sharing same columns but different time points
		Always Call merge_df_rows before calling merge_df_columns as time has
		not been set as index yet

		Arguments:
			dflist {list of pd.DataFrame} -- dataframe list to merge
		
		Returns:
			[pd.DataFrame] -- merged dataframe
		"""
		# Create Dataframe from the dlist files
		df = pd.concat(dflist, axis=0, join='outer', sort=False)

		# Sort the df based on the datetime index
		# df = df.sort_values(by=df.index)
		df = df.sort_index()

		# Dropping duplicated time points that may exist in the data
		df = df[~df.index.duplicated()]

		return df


# Methods for cleaning data
def merge_df_columns(dlist):
	"""Merge dataframes  sharing same rows but different columns
	
	Arguments:
		dlist {[list]} -- list of dataframes to be along column axis
	
	Returns:
		[pd.DataFrame] -- concatenated dataframe
	"""
	df = pd.concat(dlist, axis=1, join='outer', sort=False)
	df = dropNaNrows(df)

	return df


# Methods for cleaning data
def dropNaNrows(df):
	"""Drop rows with NaN in any column
	
	Arguments:
		df {[pd.DataFrame]} -- dataframe from which to drop NaN
	
	Returns:
		[pd.DataFrame] -- cleaned dataframe
	"""
	return df.dropna(axis=0, how='any')


# Methods for cleaning data
def dropNaNcols(df, threshold = 0.95):
	"""Drop cols with NaN > (1-threshold) fraction in any column
	
	Arguments:
		df {[pd.DataFrame]} -- dataframe from which to drop NaN
	
	Returns:
		[pd.DataFrame] -- cleaned dataframe
	"""
	return df.dropna(axis=1, thresh=int(df.shape[0]*threshold))


# Methods for cleaning data
def constantvaluecols(df, limit : Union[float, np.ndarray] = 0.02):
	"""Drop columns which are constant values and may not controibute significant information
	
	Arguments:
		df {pd.DataFrame} -- The dataframe to drop constants from
	
	Keyword Arguments:
		limit {float or 1-d np.array} -- if std less than this limit treat it as constant (default: {0.2})
	"""

	statistics = df.describe()
	if isinstance(limit, np.ndarray):
		limit = limit.flatten()
		errmsg = "Shape mismatch. Flattened limit array has length {} and number of\
			 Dataframe columns is {}".format(limit.shape[0], df.shape[1])
		assert limit.shape[0] == df.shape[1], errmsg
		constantcols = np.where(statistics.loc['std'].to_numpy()<=limit)
	else:
		constantcols = np.where(statistics.loc['std'].to_numpy()<=limit*np.ones(df.shape[1], dtype=float))
	return df.drop(columns=df.columns[constantcols])


# Methods for cleaning data
def removeoutliers(df, columns: list, **kwargs):
	"""Remove two types of outliers depending on "rmvtype" arguement
	*Remove outliers beyond z_thresh standard deviations in the data
	*Remove based on bounds
	
	Arguments:
		df {pd.DataFrame} -- the dataframe to remove outliers from
		columns {list} -- Columns to remove outliers from
	
	Keyword Arguments:
		Either -for statistical outlier removal
			z_thresh {int} -- How many standard deviations to consider for outlier removal
		Or -for bounds based outlier removal
			upperbound {float} -- upperbound for cutting off nonstatistical data
			lowerbound {float} -- lowererbound for cutting off nonstatistical data
	"""
	org_shape = df.shape[0]

	x = 'z_thresh' in kwargs.keys()
	y = 'upperbound' and 'lowerbound' in kwargs.keys()
	assert bool(~x&y | x&~y) , "Either z_thresh or both (upperbound and lowerbound) keyword arguments must be provided"


	if 'z_thresh' in kwargs.keys():  # do statistical outlier removal

		for column_name in columns:
		# Constrains will contain `True` or `False` depending on if it is a value below the threshold.
			constraints = abs(stats.zscore(df[column_name])) < kwargs['z_thresh']
			# Drop values set to be rejected
			df = df.drop(df.index[~constraints], axis = 0)

	else:  # do boundary based outlier removal

		upperbound = kwargs['upperbound']
		lowerbound = kwargs['lowerbound']
		# For every row apply threshold using bounds and see whether all columns for each row satisfy the bounds
		constraints = df.swifter.apply(lambda row: all([(cell < upperbound) and (cell > lowerbound) for cell in row[columns]]), axis=1)
		# Drop values set to be rejected
		df = df.drop(df.index[~constraints], axis = 0)

	print("Retaining {}% of the data".format(100*df.shape[0]/org_shape))

	return df


# plot a data frame
def dataframeplot(df, lazy = True, style = 'b--', ylabel : str = 'Y-axis', xlabel : str = 'X-axis', legend = False):
	"""Inspects all the rows of data in one or separate plots
	
	Arguments:
		df {pd.DataFrame} -- The dataframe to plot
	
	Keyword Arguments:
		lazy {bool} -- If true, single plot object plots all columns. Preferably set to false for plotting
		 many columns(default: {True})
		style {str} -- type of line (default: {'*'})
		ylabel {str} -- label for yaxis (default: {'Y-axis'})
		xlabel {str} -- label for xaxis (default: {'X-axis'})
		legend {bool} -- whether we want ot see legends. Turned off for many columns (default: {False})
	"""

	df_dates = df.index
	dates_num = date2num(list(df_dates))



	if not lazy:
		width, height = 20, int(df.shape[1]*3)
		plt.rcParams["figure.figsize"] = (width, height)
		font = {'family': "Times New Roman"}
		plt.rc('font', **font)
		_, ax = plt.subplots(nrows = df.shape[1], squeeze=False, constrained_layout=True)
		for i,j in zip(df.columns,range(df.shape[1])):
			#df.plot(y=[i],ax=ax[j][0],style=[style], legend=legend)
			ax[j][0].plot_date(dates_num, df[i], style, label=i)
			ax[j][0].set_xlabel(xlabel)
			ax[j][0].set_ylabel(i)
			ax[j][0].legend(prop={'size':14})
		plt.show()
	else:
		ax = df.plot(y=df.columns, figsize=(20,7), legend=legend, style = [style]*df.shape[1])
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		plt.show()
