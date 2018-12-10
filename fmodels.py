import numpy as np
from sklearn.svm import SVC
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.initializers import he_normal, he_uniform
from keras.layers.normalization import BatchNormalization
import tradingene.ind.ti as ti

def create_nn_32x16x8( num_features, num_classes=2 ):
	if isinstance(num_features,str):
		return( True ) # Returns "True" if the model requires one-hot encoding and "False" otherwise]

	m = Sequential()
	m.add(Dense(units=32, activation='tanh', input_dim=num_features, kernel_initializer=he_uniform(1)))
	m.add(Dense(16, activation='tanh'))
	m.add(Dense(8, activation='tanh'))
	m.add(Dense(num_classes, activation='softmax'))
	# Compiling the model
	m.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	return m
# end of create_model()


def create_nn( num_features, num_classes=2 ):
	if isinstance(num_features,str):
		return( True ) # Returns "True" if the model requires one-hot encoding and "False" otherwise]

	m = Sequential()
	m.add(Dense(units=num_features*4, activation='tanh', input_dim=num_features, kernel_initializer=he_uniform(1)))
	m.add(Dense(num_features*4, activation='tanh'))
	m.add(Dense(num_classes, activation='softmax'))
	# Compiling the model
	m.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	return m
# end of create_model()


def create_svm( max_iter, cw=None ):
	if isinstance(max_iter,str):
		return( False ) # Returns "True" if the model requires one-hot encoding and "False" otherwise]

	m = SVC(gamma='auto', kernel='poly', probability=True, verbose=True, tol=1e-3, degree=5, max_iter=max_iter, class_weight='balanced')
	return m
# end of create_model()


def calculate_input_r(candles):
	if isinstance(candles,str):
		return( 10, 8 ) # Returning 1) a lookback period involved and 2) the number of features generated

	input_vec = []
	counter=1  # 1, 3 (3), 6 (4), 10 (5)
	counter_increment=2
	while True:
		meanh = np.mean(candles['high'][:counter])
		meanl = np.mean(candles['low'][:counter])
		input_vec.append( meanh )
		input_vec.append( meanl )
		counter += counter_increment
		if counter > 10:
			break
		counter_increment += 1
	
	return(input_vec)	
# end of ...

def calculate_input_i(candles):
	if isinstance(candles,str):
		return( 15, 20 ) # Returning 1) a lookback period involved and 2) the number of features generated

	input_vec = []
	for i in range(4): # Returns
		input_vec.append( (candles['open'][i] - candles['close'][i]) / candles['close'][i] )
	for i in range(4): # sma
		sma = ti.sma( 10, i, candles['close'] )			
		input_vec.append( (sma - candles['close'][0]) / candles['close'][0] )
	for i in range(4): # rsi
		rsi = ti.rsi( 10, i, candles['close'] )			
		input_vec.append( rsi['rsi'] )
	for i in range(4): # rsi
		momentum = ti.momentum(10, i, candles['close'] )			
		input_vec.append( momentum )
	for i in range(4): # volumes
		input_vec.append( candles['vol'][i] )
	return(input_vec)
# end of calculate_input()
