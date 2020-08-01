"""
This script will implement the data driven models that will be used as part of the
simulator environment
"""

# import
from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, classification_report, roc_auc_score

import warnings
with warnings.catch_warnings():
	import tensorflow as tf
	from keras import backend as K
	from keras.models import Model
	from keras.layers import Input, Dense, LSTM, Reshape
	from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

class datadrivenmodel():
	"""
	This class would implement the different methods to learn the different data 
	driven models
	"""


	def __init__(self, *args,**kwargs):
		"""
		Initiate the class with necessary parameters and variables to be used later
		"""
		raise NotImplementedError


	def design(self, *args, **kwargs):
		"""
		Create the network structure; pass args,kwargs if needed
		"""
		raise NotImplementedError


	def fit(self, *args, **kwargs):
		"""
		Train the model; pass args,kwargs if needed
		"""
		raise NotImplementedError

	def predict(self, *args, **kwargs):
		"""
		Prediction from the model on some test data; pass args,kwargs if needed
		"""
		raise NotImplementedError


class nn_model(datadrivenmodel):
	"""
	Creates the data driven model for predicting energy
	"""

	def __init__(self, *args, **kwargs):
		"""
		Initiate the class
		"""
		self.model_type = kwargs['model_type']
		self.graph = tf.Graph()
		self.session = tf.Session(graph=self.graph)
		self.train_batchsize = kwargs['train_batchsize']
		self.input_timesteps = kwargs['input_timesteps']
		self.input_dim = kwargs['input_dim']
		self.save_path = kwargs['save_path']
		self.model_path = kwargs['model_path']
		self.timegap = kwargs['timegap']*5
		self.epochs = 0
		self.initial_epoch = 0
		self.name = kwargs['name']


		if self.model_type == 'regresion':
			self.outputdim = 1
			self.loss = 'mse'
			self.last_activation = 'relu'
		elif self.model_type == 'classification':
			self.outputdim = 2
			self.loss = 'binary_crossentropy'
			self.last_activation = 'softmax'
		else:
			raise ValueError('model_type has to be either regression or classification')


	def design(self, *args, **kwargs):
		"""
		Design the network
		"""
		dense_layers, dense_units = kwargs['dense_layers'], kwargs['dense_units']
		activation_dense = kwargs['activation_dense']
		lstm_layers, lstm_units = kwargs['lstm_layers'], kwargs['lstm_units']
		activation_lstm = kwargs['activation_lstm']
		self.lstm_layer_num =  kwargs['lstm_layers']

		with self.graph.as_default():  # pylint: disable=not-context-manager
			with self.session.as_default():  # pylint: disable=not-context-manager
				input_layer = Input(batch_shape=(None, self.input_timesteps, self.input_dim), name = kwargs['name']+'_input')
				layers = input_layer
				for i in range(dense_layers):
					layers = Dense(dense_units, activation=activation_dense, name=kwargs['name']+'_dense'+str(i))(layers)
				for i in range(lstm_layers-1):
					layers = LSTM(lstm_units, activation=activation_lstm, return_sequences=True,
					 name=kwargs['name']+'_lstm'+str(i))(layers)
				output = LSTM(self.outputdim, activation=self.last_activation, return_sequences=False,
					 name=kwargs['name']+'_lstm'+str(i+1))(layers)
				output = Reshape((1,self.outputdim), name = kwargs['name']+'_reshape')(output)
				self.model = Model(inputs=input_layer, outputs=output)


	def fit(self, *args, **kwargs):
		"""
		Fit the model to the data
		"""
		# train the model
		with self.graph.as_default():  # pylint: disable=not-context-manager
			with self.session.as_default():  # pylint: disable=not-context-manager
				self.history = self.model.fit(kwargs['X_train'], kwargs['y_train'],
											validation_data=(kwargs['X_val'], kwargs['y_val']),
											batch_size=self.train_batchsize,
											epochs=kwargs['epochs'], initial_epoch =self.initial_epoch,
											callbacks=self.callbacks(), verbose=0, shuffle=False,)
		try:
			self.initial_epoch += len(self.history.history['loss'])
		except KeyError:
			pass

	
	def predict(self, *args, **kwargs):
		"""
		Evaluate the model; pass args,kwargs if needed
		"""
		with self.graph.as_default():  # pylint: disable=not-context-manager
			with self.session.as_default():  # pylint: disable=not-context-manager
				predictions = self.model.predict(kwargs['X_test'])
		return predictions
	
	
	def compile(self, *args, **kwargs):
		"""
		Compile the network
		"""
		with self.graph.as_default():  # pylint: disable=not-context-manager
			with self.session.as_default():  # pylint: disable=not-context-manager
				self.model.compile(loss=self.loss,optimizer='adam')


	def callbacks(self, *args, **kwargs):
		"""
		Create callbacks
		"""
		self.modelchkpt = ModelCheckpoint(self.model_path +self.name+'_best_model',
			monitor = 'val_loss', save_best_only = True, period=2)
		self.earlystopping = EarlyStopping(monitor = 'val_loss', patience=5, restore_best_weights=False)
		self.reduclronplateau = ReduceLROnPlateau(monitor = 'val_loss', patience=2, cooldown = 3)
		self.tbCallBack = TensorBoard(log_dir=self.save_path+'loginfo', batch_size=self.train_batchsize, histogram_freq=0,
		write_graph=False, write_images=False, write_grads=True)
		self.cb_list = [self.modelchkpt, self.earlystopping, self.reduclronplateau, self.tbCallBack]
		return self.cb_list


	def re_init_layers(self, *args, **kwargs):
		"""
		Freeze certain layers and reinitialize certain layers
		"""
		with self.graph.as_default():  # pylint: disable=not-context-manager
			with self.session.as_default():  # pylint: disable=not-context-manager
				for layer in self.model.layers[1:-1]:
					if 'cell' in layer.__dict__.keys():
						layer.cell.kernel.initializer.run(session=K.get_session())
					else:
						layer.kernel.initializer.run(session=K.get_session())
				self.model.compile(loss=self.loss,optimizer='adam')

	def load_weights(self, *args, **kwargs):
		"""
		Load weights from the given path
		"""
		with self.graph.as_default():  # pylint: disable=not-context-manager
			with self.session.as_default():  # pylint: disable=not-context-manager
				self.model.load_weights(self.model_path +self.name+'_best_model')
				self.model.compile(loss=self.loss,optimizer='adam')


class no_val_nn_model(datadrivenmodel):
	"""
	Creates the data driven model for predicting energy
	"""

	def __init__(self, *args, **kwargs):
		"""
		Initiate the class
		"""
		self.model_type = kwargs['model_type']
		self.graph = tf.Graph()
		self.session = tf.Session(graph=self.graph)
		self.train_batchsize = kwargs['train_batchsize']
		self.input_timesteps = kwargs['input_timesteps']
		self.input_dim = kwargs['input_dim']
		self.save_path = kwargs['save_path']
		self.model_path = kwargs['model_path']
		self.timegap = kwargs['timegap']*5
		self.epochs = 0
		self.initial_epoch = 0
		self.name = kwargs['name']


		if self.model_type == 'regresion':
			self.outputdim = 1
			self.loss = 'mse'
			self.last_activation = 'relu'
		elif self.model_type == 'classification':
			self.outputdim = 2
			self.loss = 'binary_crossentropy'
			self.last_activation = 'softmax'
		else:
			raise ValueError('model_type has to be either regression or classification')


	def design(self, *args, **kwargs):
		"""
		Design the network
		"""
		dense_layers, dense_units = kwargs['dense_layers'], kwargs['dense_units']
		activation_dense = kwargs['activation_dense']
		lstm_layers, lstm_units = kwargs['lstm_layers'], kwargs['lstm_units']
		activation_lstm = kwargs['activation_lstm']
		self.lstm_layer_num =  kwargs['lstm_layers']

		with self.graph.as_default():  # pylint: disable=not-context-manager
			with self.session.as_default():  # pylint: disable=not-context-manager
				input_layer = Input(batch_shape=(None, self.input_timesteps, self.input_dim), name = kwargs['name']+'_input')
				layers = input_layer

				for i in range(lstm_layers-1):
					layers = LSTM(lstm_units, activation=activation_lstm, return_sequences=True,
					 name=kwargs['name']+'_lstm'+str(i))(layers)

				layers = LSTM(lstm_units, activation=activation_lstm, return_sequences=False,
					 name=kwargs['name']+'_lstm'+str(i+1))(layers)

				for j in range(dense_layers-1):
					layers = Dense(dense_units, activation=activation_dense, name=kwargs['name']+'_dense'+str(j))(layers)
				
				layers = Dense(self.outputdim, activation=self.last_activation, name=kwargs['name']+'_dense'+str(j+1))(layers)

				output = Reshape((1, self.outputdim), name = kwargs['name']+'_reshape')(layers)
				
				self.model = Model(inputs=input_layer, outputs=output)


	def fit(self, *args, **kwargs):
		"""
		Fit the model to the data
		"""
		# train the model
		with self.graph.as_default():  # pylint: disable=not-context-manager
			with self.session.as_default():  # pylint: disable=not-context-manager
				self.history = self.model.fit(kwargs['X_train'], kwargs['y_train'],
											batch_size=self.train_batchsize,
											epochs=kwargs['epochs'], initial_epoch =self.initial_epoch,
											callbacks=self.callbacks(), verbose=0, shuffle=False,)
		try:
			self.initial_epoch += len(self.history.history['loss'])
		except KeyError:
			pass

	
	def predict(self, *args, **kwargs):
		"""
		Evaluate the model; pass args,kwargs if needed
		"""
		with self.graph.as_default():  # pylint: disable=not-context-manager
			with self.session.as_default():  # pylint: disable=not-context-manager
				predictions = self.model.predict(kwargs['X_test'])
		return predictions
	
	
	def compile(self, *args, **kwargs):
		"""
		Compile the network
		"""
		with self.graph.as_default():  # pylint: disable=not-context-manager
			with self.session.as_default():  # pylint: disable=not-context-manager
				self.model.compile(loss=self.loss,optimizer='adam')


	def callbacks(self, *args, **kwargs):
		"""
		Create callbacks
		"""
		self.modelchkpt = ModelCheckpoint(self.model_path +self.name+'_best_model',
			monitor = 'loss', save_best_only = True, period=2)
		self.earlystopping = EarlyStopping(monitor = 'loss', patience=5, restore_best_weights=False)
		self.reduclronplateau = ReduceLROnPlateau(monitor = 'loss', patience=2, cooldown = 3)
		self.tbCallBack = TensorBoard(log_dir=self.save_path+'loginfo', batch_size=self.train_batchsize, histogram_freq=0,
		write_graph=False, write_images=False, write_grads=True)
		self.cb_list = [self.modelchkpt, self.earlystopping, self.reduclronplateau, self.tbCallBack]
		return self.cb_list


	def re_init_layers(self, *args, **kwargs):
		"""
		Freeze certain layers and reinitialize certain layers
		"""
		with self.graph.as_default():  # pylint: disable=not-context-manager
			with self.session.as_default():  # pylint: disable=not-context-manager
				for layer in self.model.layers[1:-1]:
					if 'cell' in layer.__dict__.keys():
						layer.cell.kernel.initializer.run(session=K.get_session())
					else:
						layer.kernel.initializer.run(session=K.get_session())
				self.model.compile(loss=self.loss,optimizer='adam')

	def load_weights(self, *args, **kwargs):
		"""
		Load weights from the given path
		"""
		with self.graph.as_default():  # pylint: disable=not-context-manager
			with self.session.as_default():  # pylint: disable=not-context-manager
				self.model.load_weights(self.model_path +self.name+'_best_model')
				self.model.compile(loss=self.loss,optimizer='adam')


def regression_evaluate(prediction: np.ndarray, target: np.ndarray, **kwargs):

	assert prediction.shape == target.shape, "Prediction {} and Target {} \
		arrays are of different shape".format(prediction.shape, target.shape)

	# log error on test data
	rmse = sqrt(mean_squared_error(target, prediction))
	cvrmse = 100*(rmse/np.mean(prediction))
	mae = mean_absolute_error(target, prediction)
	file = open(kwargs['save_path'] + kwargs['timegap']+' min(s) results.txt','a')
	file.write('{}-Time Step {}: Test RMSE={} |Test CVRMSE={} |Test MAE={}\n'.format(kwargs['Idx'], 1, rmse, cvrmse, mae))
	file.close()


def classification_evaluate(prediction: np.ndarray, target: np.ndarray, **kwargs):

	prediction_class = np.argmax(prediction, axis=-1)
	target = np.argmax(target, axis=-1)

	results = classification_report(target.flatten(), prediction_class.flatten(), output_dict=True)
	pred_score = np.choose(prediction_class.flatten(),prediction.T).flatten()
	try:
		roc_score = roc_auc_score(target.flatten(), pred_score)
	except ValueError:
		roc_score = 0.5

	file = open(kwargs['save_path'] + kwargs['timegap']+' min(s) results.txt','a')
	file.write('{}: Test Accuracy={} |Test Precision={} |Test ROC={}\n'.format(kwargs['Idx'],
		results['accuracy'], results['weighted avg']['precision'], roc_score))
	file.close()
