import sys
if len(sys.argv) < 2:
	print( "Use: filtering.py <a-file-with-trades.csv>" )
	sys.exit()

from datetime import datetime, timedelta
from tradingene.data.load import import_data, import_candles
from tradingene.algorithm_backtest.tng import TNG
import tradingene.backtest_statistics.backtest_statistics as bs
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.initializers import he_normal, he_uniform
from keras.layers.normalization import BatchNormalization
import tradingene.ind.ti as ti
import keras
import numpy as np
import utils
import matplotlib.pyplot as plt 
from random import randint
from operator import itemgetter
import re 

# 17-5-2-200-120-10-0.01-0.85-3
THRESHOLD_ABS = 50
THRESHOLD_PCT = 1.45
THRESHOLD_TRADES_NUMBER = 50

PREDICTION_THRESHOLD = 1.25

PERIOD_OF_INDICATORS = 10
LOOKBACK = 4
LOOKBACK_CANDLES = LOOKBACK + PERIOD_OF_INDICATORS
NUM_FEATURES = 20 # The number of features. This depends on how you implement the "calculate_input()" function. 
#LOOKBACK_CANDLES = 15
#NUM_FEATURES = 5
num_classes = 2 # The number of classes depends of what the "calculate_output()" fuction returns (see below)
EPOCHS = 1000 # The number of EPOCHS to train your model through.
TIMEFRAME = 30 # The time frame.
TICKER = "btcusd" # The ticker.
TRAIN_VS_TEST = 0.75
SRC_FORMAT = 'platform'

for a in range(2, len(sys.argv)): # Reading additional parameters
	m = re.match(r'([a-zA-Z0-9\_ ]+=[0-9a-zA-Z \.\'\_]+)', sys.argv[a])
	if m:
		exec(m.group(1))

def load_data( src_format, ticker, timeframe, calculate_input_fn, lookback_candles, threshold_abs, threshold_pct, \
            train_vs_test = 0.75, use_weights_for_classes = False ):
	
    data['train_inputs'] = []
	data['train_outputs'] = []
	data['train_profit'] = []
	data['train_profit_pct'] = []
	data['train_trade_num'] = []
	data['test_inputs'] = []
	data['test_outputs'] = []
	data['test_profit'] = []
	data['test_profit_pct'] = []
	data['test_trade_num'] = []

	if src_format == 'platform': 
		trades = utils.read_trades(sys.argv[1])
	else:
		trades = utils.read_trades_2(sys.argv[1])
	if len(trades) == 0:
		return None

	trades = utils.trim_trades( trades )
	if len(trades) == 0:
		return None

	# Searching for min & max... 
	dt_min = trades[0]['time_dt']
	dt_max = trades[0]['closing_time_dt']
	for i in range( 1, len(trades) ):
		if trades[i]['time_dt'] < dt_min:
			dt_min = trades[i]['time_dt']
		if trades[i]['closing_time_dt'] > dt_max:
			dt_max = trades[i]['closing_time_dt']

	# Calculating train and test starting and ending dates
	dt_start = dt_min - timedelta(minutes = timeframe * (lookback_candles + 1))
	dt_end = dt_max + timedelta(minutes=timeframe*2)
	num_days = (dt_max - dt_min).days
	num_days_in_test = int(num_days * train_vs_test)
	dt_train_end = dt_min + timedelta(num_days_in_test)

	sys.stderr.write( 'DATES INVOLVED: start: %s, end: %s, train end: %s' % ( str(dt_start), str(dt_end), str(dt_train_end) ) )

	# Importing candles...
	candles = import_candles(ticker, timeframe, dt_start, dt_end)

	utils.merge_candles_and_trades( candles, trades )
	utils.calculate_inputs( candles, trades, data, calculate_input_fn, lookback_candles )

	# Searching for "good" and "bad" trades... 
	num_bad = 0
	num_good = 0
	for t in range(len(data['trade_num'])):
		profit = data['profit'][t]
		profit_pct = data['profit_pct'][t]
		if profit > threshold_abs and profit_pct > threshold_pct:
			data['outputs'][t] = [0,1]
			num_good += 1
		else:
			data['outputs'][t] = [1,0]
			num_bad += 1

	print('num_good=%d, num_bad=%d' % (num_good,num_bad) )

	num_classes = int( float(num_bad) / float(num_good) + 0.9 ) + 1
	print("num_classes=%d"%(num_classes))
	if num_classes > 1:
        if not use_weight_for_classes:
    		trades_sorted = [ x for _, x in sorted(zip( data['profit_pct'], data['trade_num'] )) ] # Trade numbers sorted by profit 
    		for b in range(num_classes): # For each bin 
    			startIndex = int( b * (num_good + num_bad) / (num_classes) ) # Starting and ending indexes within the trades_sorted array
    			endIndex = int( (b+1) * (num_good + num_bad) / (num_classes) )

    			output = [] # Assigning a "one-hot" output with the right significant bin... 
    			for i in range( num_classes ):
    				if i == b:
    					output.append(1)
    				else:
    					output.append(0)

    			for i in range( startIndex, endIndex ):
    				trade_num = trades_sorted[i] #
    				for t in range(len(data['trade_num'])): # Searching for train data related to "trade_num" trade...
    					if data['trade_num'][t] == trade_num:
    						data['outputs'][t] = output
    						break
	else:
		print("ACCORDING TO THE SETTINGS SPECIFIED THERE IS ONLY ONE CLASS IN THE SAMPLE SET. Exiting...")
		sys.exit();
	# Splitting for train and test
	for t in range(len(data['time_dt'])):
		if data['time_dt'][t] < dt_train_end:
			data['train_inputs'].append(data['inputs'][t])
			data['train_outputs'].append(data['outputs'][t])
			data['train_profit'].append(data['profit'][t])
			data['train_profit_pct'].append(data['profit_pct'][t])
			data['train_trade_num'].append(data['trade_num'][t])
		else:
			data['test_inputs'].append(data['inputs'][t])
			data['test_outputs'].append(data['outputs'][t])
			data['test_profit'].append(data['profit'][t])
			data['test_profit_pct'].append(data['profit_pct'][t])
			data['test_trade_num'].append(data['trade_num'][t])
	# end of "for"

	data['train_inputs'] = np.array(data['train_inputs'])
	data['train_outputs'] = np.array(data['train_outputs'])
	data['test_inputs'] = np.array(data['test_inputs'])
	data['test_outputs'] = np.array(data['test_outputs'])

	scaler = StandardScaler() # Creating an instance of a scaler.
	scaler.fit(data['train_inputs']) # Fitting the scaler.
	data['train_inputs'] = scaler.transform(data['train_inputs']) # Normalizing data

	return {'data':data, 'candles':candles , 'trades':trades, 'scaler':scaler, 
        'num_classes':num_classes, 'num_good':num_good, 'num_bad':num_bad, }
# end of load_data


def load_data_and_trainmodel():
	global num_models, models

	if not load_data():
		print("Can't load data! Exiting...")
		sys.exit();

	inputs = data['train_inputs']
	outputs = data['train_outputs']
	model = create_model() # Creating a model
    if not USE_WEIGHTS_FOR_CLASSES:
	   model.fit( inputs, outputs, epochs=EPOCHS ) # Training the model
    else:
        model.fit( inputs, outputs, epochs=EPOCHS )
	models.append( model )
	num_models = 1


def create_model():
	m = Sequential()
	# The number of nodes in the first hidden layer equals the number of features
	m.add(Dense(units=32, activation='tanh', input_dim=NUM_FEATURES, kernel_initializer=he_uniform(1)))
	#m.add(Dense(units=NUM_FEATURES*4, activation='tanh', input_dim=NUM_FEATURES, kernel_initializer=he_uniform(1)))
	# Adding another hidden layer 
	m.add(Dense(16, activation='tanh'))
	m.add(Dense(8, activation='tanh'))
	# m.add(Dense(NUM_FEATURES*4, activation='tanh'))
	# Adding an output layer
	m.add(Dense(num_classes, activation='softmax'))
	# Compiling the model
	m.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	return m
# end of create_model()


def calculate_input(candles):

	input_vec = []

	global NUM_FEATURES, LOOKBACK_CANDLES
	NUM_FEATURES = 5
	LOOKBACK_CANDLES = 5
	for i in range(NUM_FEATURES): # Returns
		input_vec.append( (candles['close'][0] - candles['open'][i]) / candles['close'][0] )
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

load_data_and_trainmodel() 

(all_trades,train_trades,test_trades) = utils.calc_trades_in_every_bin(data,num_classes)
print("ALL, TRAIN, TEST")
print(all_trades)
print(train_trades)
print(test_trades)
input("PRESS ENTER")

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
	inp = scaler.transform(inp)		
	trade_num = data['test_trade_num'][t]
	profit = trades[trade_num]['profit']
	num_models_that_allowed_trade = 0

	for m in range(num_models): # Iterating through all the models trained to obtain predictions
		prediction = models[m].predict_proba( inp )[0]
		print(prediction)
		print(data['test_outputs'][t])
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
		if data['test_outputs'][t][num_classes-1] == 1:
			oked_right += 1
		else:
			oked_wrong += 1

	print("Trades (actual/oked): %d (%d), oked right: %d, oked wrong: %d, profit_actual=%f, profit_optimized=%f" % \
		(num_trades, num_trades_oked, oked_right, oked_wrong, profit_actual, profit_optimized))

pnl = {}
utils.calculate_pnl(candles, trades_actual, pnl, "trades_actual_")
utils.calculate_pnl(candles, trades_oked, pnl, "trades_oked_")

import matplotlib.pyplot as plt
plt.plot(pnl['time_dt'], pnl['trades_actual_pnl'])
plt.plot(pnl['time_dt'], pnl['trades_oked_pnl'])
plt.show()

