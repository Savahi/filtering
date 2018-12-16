import sys
if len(sys.argv) < 2:
	sys.stderr.write( "Use: filtering.py <a-file-with-trades.csv>" )
	sys.exit()

from datetime import datetime, timedelta
import re, numpy as np, matplotlib.pyplot as plt 
import futils, fmodels
#from tradingene.algorithm_backtest.tng import TNG
#import tradingene.backtest_statistics.backtest_statistics as bs

THRESHOLD_ABS = 0.0
THRESHOLD_PCT = 1.0
THRESHOLD_TRADES_NUMBER = 50
PREDICTION_THRESHOLD = 1.00001
NUM_MODELS = 1
USE_WEIGHTS_FOR_CLASSES = False # Makes 2 classes with weights assigned accordingly to number of samples in each class
USE_2_CLASSES_WITH_RANDOM_PICK = False # Make two classes and several models 
NUM_EPOCHS = 1000 # The number of EPOCHS to train your model through.
TIMEFRAME = 15 # The time frame.
TICKER = "btcusd" # The ticker.
TRAIN_VS_TEST = None
NUM_CLASSES = None
SRC_FORMAT = 'platform'
MODEL_TYPE = 'nn'
CREATE_MODEL = fmodels.create_nn_32x16x8  # fmodels.create_nn
CALCULATE_INPUT = fmodels.calculate_input_r # fmodels.calculate_input_i
PRINT_HEADER = False
VERBOSE = True
COMMISSION_PCT=0.0
COMMISSION_ABS=0.0
FILTER_BY_WORST_CLASS = False

if re.search( r'\.lst$', sys.argv[1] ): # If a source file passed through command line ends with ".lst"...
	sys.stderr.write('Reading list of source files...\n')
	src = futils.read_list_of_files_with_trades( sys.argv[1] )
	if len(src) == 0:
		sys.stderr.write('Failed to load list of source files. Exiting...\n')
		sys.exit(0)
else: # A source passed though command line is a file name 
	src = sys.argv[1]

command_line_arguments = ''
for a in range(2, len(sys.argv)): # Reading additional parameters
	m = re.match(r'([a-zA-Z0-9\_ ]+=[0-9a-zA-Z \.\'\"\_]+)', sys.argv[a])
	if m:
		exec(m.group(1))
	command_line_arguments += ' ' + sys.argv[a]

res = futils.load_trades_and_candles( src, SRC_FORMAT, TICKER, TIMEFRAME, 
	extra_lookback_candles=CALCULATE_INPUT('query_input')[0], commission_pct=COMMISSION_PCT, commission_abs=COMMISSION_ABS )
if res is None:
	sys.stderr.write('Failed to load trades or candles. Exiting...\n')
	sys.exit(0)
candles = res['candles']
trades = res['trades']

res = futils.calculate_data_and_train_models(candles, trades, CREATE_MODEL, CALCULATE_INPUT,
	threshold_abs=THRESHOLD_ABS, threshold_pct=THRESHOLD_PCT, train_vs_test=TRAIN_VS_TEST, num_classes=NUM_CLASSES,
	use_weights_for_classes=USE_WEIGHTS_FOR_CLASSES, use_2_classes_with_random_pick=USE_2_CLASSES_WITH_RANDOM_PICK, 
	num_models=NUM_MODELS, num_epochs=NUM_EPOCHS, verbose=VERBOSE )
data=res['data']
(all_trades,train_trades,test_trades) = futils.calc_trades_in_every_bin(data, res['num_classes'])
sys.stderr.write("ALL, TRAIN, TEST\n")
sys.stderr.write("%s, %s, %s\n" % (str(all_trades), str(train_trades), str(test_trades)))
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

	if not FILTER_BY_WORST_CLASS:
		filter_class = num_classes-1
	else:
		filter_class = 0
	num_models_that_confirmed_filter_class = 0

	for m in range(num_models): # Iterating through all the models trained to obtain predictions
		prediction = models[m].predict_proba( inp )[0]
		sys.stderr.write(str(prediction) + "\n")
		sys.stderr.write(str(data['test_outputs'][t]) + "\n")
		if np.argmax(prediction) == filter_class: 
			if prediction[filter_class] > PREDICTION_THRESHOLD * 1.0/num_classes:
				num_models_that_confirmed_filter_class += 1
	
	profit_actual += profit
	num_trades += 1

	trades_actual.append(trades[trade_num])

	oked = False
	if not FILTER_BY_WORST_CLASS: # Filtering by best class
		if num_models_that_confirmed_filter_class == num_models:
			oked = True
	else: # Filtering by worst class
		if num_models_that_confirmed_filter_class < num_models:
			oked = True

	if oked: # If oked...
		trades_oked.append(trades[trade_num])
		profit_optimized += profit
		num_trades_oked += 1			
		if MODEL_TYPE == 'nn':
			right = data['test_outputs'][t][filter_class] == 1
		else:
			right = data['test_outputs'][t] == filter_class

		if right:
			oked_right += 1
		else:
			oked_wrong += 1

	sys.stderr.write("Trades (actual/oked): %d (%d), oked right: %d, oked wrong: %d, profit_actual=%f, profit_optimized=%f\n" % \
		(num_trades, num_trades_oked, oked_right, oked_wrong, profit_actual, profit_optimized))

if PRINT_HEADER:
	print("File, Num. of Trades, Num. of Trades Allowed, +, -, Profit, Optimized Profit, CMA")
print("%s, %d, %d, %d, %d, %f, %f %s" % (sys.argv[1], num_trades, num_trades_oked, oked_right, oked_wrong, profit_actual, profit_optimized, command_line_arguments))

pnl = {}
futils.calculate_pnl(candles, trades_actual, pnl, "trades_actual_")
futils.calculate_pnl(candles, trades_oked, pnl, "trades_oked_")

import matplotlib.pyplot as plt
plt.plot(pnl['time_dt'], pnl['trades_actual_pnl'])
plt.plot(pnl['time_dt'], pnl['trades_oked_pnl'])
plt.savefig( sys.argv[1] + '.png')

