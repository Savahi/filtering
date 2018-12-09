import sys
if len(sys.argv) < 2:
	sys.stderr.write( "Use: filtering.py <a-file-with-trades.csv>" )
	sys.exit()

from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt 
import utils
import re 
from sklearn.svm import SVC
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.initializers import he_normal, he_uniform
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import StandardScaler
from tradingene.data.load import import_data, import_candles
from tradingene.algorithm_backtest.tng import TNG
import tradingene.backtest_statistics.backtest_statistics as bs
import tradingene.ind.ti as ti

THRESHOLD_ABS = 0.0
THRESHOLD_PCT = 1.0
THRESHOLD_TRADES_NUMBER = 50
PREDICTION_THRESHOLD = 1.25
NUM_MODELS = 1
USE_WEIGHTS_FOR_CLASSES = False # Makes 2 classes with weights assigned accordingly to number of samples in each class
USE_2_CLASSES_WITH_RANDOM_PICK = False # Make two classes and several models 
NUM_EPOCHS = 1000 # The number of EPOCHS to train your model through.
TIMEFRAME = 30 # The time frame.
TICKER = "btcusd" # The ticker.
TRAIN_VS_TEST = 0.75
SRC_FORMAT = 'platform'
MODEL_TYPE = 'nn'

for a in range(2, len(sys.argv)): # Reading additional parameters
	m = re.match(r'([a-zA-Z0-9\_ ]+=[0-9a-zA-Z \.\'\_]+)', sys.argv[a])
	if m:
		exec(m.group(1))


def create_nn_model( num_features, num_classes ):
	m = Sequential()
	# The number of nodes in the first hidden layer equals the number of features
	m.add(Dense(units=32, activation='tanh', input_dim=num_features, kernel_initializer=he_uniform(1)))
	#m.add(Dense(units=num_features*4, activation='tanh', input_dim=num_features, kernel_initializer=he_uniform(1)))
	# Adding another hidden layer 
	m.add(Dense(16, activation='tanh'))
	m.add(Dense(8, activation='tanh'))
	# m.add(Dense(num_features*4, activation='tanh'))
	# Adding an output layer
	m.add(Dense(num_classes, activation='softmax'))
	# Compiling the model
	m.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	return m
# end of create_model()

def create_svm_model( max_iter, cw=None ):
	m = SVC(gamma='auto', probability=True, verbose=True, tol=1e-4, max_iter=max_iter, class_weight=cw)
	return m
# end of create_model()


def calculate_input(candles):

	if isinstance(candles,str):
		return( 5, 10 ) # Returning a lookback period used and the number of features generated

	#print('len(candles)=%d' % len(candles))

	input_vec = []
	for i in range(5): # Returns
		input_vec.append( (candles['close'][0] - candles['high'][i]) * 100.0 / candles['close'][0] )
		input_vec.append( (candles['close'][0] - candles['low'][i]) * 100.0 / candles['close'][0] )
	return(input_vec)	

	for i in range(4): # Returns
		input_vec.append( (candles['open'][i] - candles['close'][i]) / candles['close'][i] )
	for i in range(4): # sma
		sma = ti.sma( PERIOD_OF_INDICATORS, i, candles['close'] )			
		input_vec.append( (sma - candles['close'][0]) / candles['close'][0] )
	for i in range(4): # rsi
		rsi = ti.rsi( PERIOD_OF_INDICATORS, i, candles['close'] )			
		input_vec.append( rsi['rsi'] )
	for i in range(4): # rsi
		momentum = ti.momentum( PERIOD_OF_INDICATORS, i, candles['close'] )			
		input_vec.append( momentum )
	for i in range(4): # volumes
		input_vec.append( candles['vol'][i] )
	return(input_vec)
# end of calculate_input()


res = utils.load_trades_and_candles( sys.argv[1], SRC_FORMAT, TICKER, TIMEFRAME, calculate_input('query_input')[0] )
if res is None:
	sys.stderr.write('Failed to load trades or candles. Exiting...\n')
	sys.exit(0)
candles = res['candles']
trades = res['trades']

if MODEL_TYPE == 'nn':
	cm = create_nn_model
else:
	cm = create_svm_model

res = utils.calculate_data_and_train_models(candles, trades, cm, calculate_input,
	threshold_abs=THRESHOLD_ABS, threshold_pct=THRESHOLD_PCT, train_vs_test=TRAIN_VS_TEST, 
	use_weights_for_classes=USE_WEIGHTS_FOR_CLASSES, use_2_classes_with_random_pick=USE_2_CLASSES_WITH_RANDOM_PICK, 
	num_models=NUM_MODELS, num_epochs=NUM_EPOCHS, model_type=MODEL_TYPE )
data=res['data']
(all_trades,train_trades,test_trades) = utils.calc_trades_in_every_bin(data, res['num_classes'])
sys.stderr.write("ALL, TRAIN, TEST\n")
sys.stderr.write("%s, %s, %s\n" % (str(all_trades), str(train_trades), str(test_trades)))
input("")
models = res['models']
num_models = len(models)
num_classes = res['num_classes']

num_trades = 0
num_trades_oked = 0
oked_right = 0
oked_wrong = 0
profit_actual = 0
profit_optimized = 0
trades_actual = []
trades_oked = []

for t in range(len(data['test_inputs'])):
	inp = np.array( [ data['test_inputs'][t] ] )
	#inp = scaler.transform(inp)		
	trade_num = data['test_trade_num'][t]
	profit = trades[trade_num]['profit']
	num_models_that_allowed_trade = 0

	for m in range(num_models): # Iterating through all the models trained to obtain predictions
		prediction = models[m].predict_proba( inp )[0]
		sys.stderr.write(str(prediction) + "\n")
		sys.stderr.write(str(data['test_outputs'][t]) + "\n")
		if np.argmax(prediction) == num_classes-1: 
			if prediction[num_classes-1] > PREDICTION_THRESHOLD * 1.0/num_classes:
				num_models_that_allowed_trade += 1
	
	profit_actual += profit
	num_trades += 1

	trades_actual.append(trades[trade_num])

	if num_models_that_allowed_trade == num_models:
		trades_oked.append(trades[trade_num])

		profit_optimized += profit
		num_trades_oked += 1			
		if MODEL_TYPE == 'nn':
			right = data['test_outputs'][t][num_classes-1] == 1
		else:
			right = data['test_outputs'][t] == num_classes - 1

		if right:
			oked_right += 1
		else:
			oked_wrong += 1

	sys.stderr.write("Trades (actual/oked): %d (%d), oked right: %d, oked wrong: %d, profit_actual=%f, profit_optimized=%f\n" % \
		(num_trades, num_trades_oked, oked_right, oked_wrong, profit_actual, profit_optimized))

print("%d, %d, %d, %d, %f, %f" % (num_trades, num_trades_oked, oked_right, oked_wrong, profit_actual, profit_optimized))

pnl = {}
utils.calculate_pnl(candles, trades_actual, pnl, "trades_actual_")
utils.calculate_pnl(candles, trades_oked, pnl, "trades_oked_")

import matplotlib.pyplot as plt
plt.plot(pnl['time_dt'], pnl['trades_actual_pnl'])
plt.plot(pnl['time_dt'], pnl['trades_oked_pnl'])
plt.savefig( sys.argv[1] + '.png')
