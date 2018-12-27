import sys
if len(sys.argv) < 2:
	sys.stderr.write( "Use: filtering.py <a-file-with-trades.csv>" )
	sys.exit()

from datetime import datetime, timedelta
import re, numpy as np, matplotlib.pyplot as plt 
import futils, fmodels
from f2sutils import test_models
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
NUM_CLASSES = 3
SRC_FORMAT = 'nn'
MODEL_TYPE = 'nn'
CREATE_MODEL = fmodels.create_nn_32x16x8  # fmodels.create_nn
CALCULATE_INPUT = fmodels.calculate_input_r # fmodels.calculate_input_i
PRINT_HEADER = False
VERBOSE = False
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
	num_models=NUM_MODELS, num_epochs=NUM_EPOCHS, verbose=VERBOSE, filter_by_side=1)
if res is not None:
	data=res['data']
	(all_trades,train_trades,test_trades) = futils.calc_trades_in_every_bin(data, res['num_classes'])
	sys.stderr.write("LONG: ALL, TRAIN, TEST\n")
	sys.stderr.write("%s, %s, %s\n" % (str(all_trades), str(train_trades), str(test_trades)))
	sys.stderr.write("Testing LONG:\n")
	test_models( candles, trades, data, res['models'], res['num_classes'], 
		model_type=MODEL_TYPE, filter_by_worst_class=FILTER_BY_WORST_CLASS, print_header=PRINT_HEADER, 
		prediction_threshold=PREDICTION_THRESHOLD, side=1)
else:
	sys.stderr.write('Not enough data!\n')
res = futils.calculate_data_and_train_models(candles, trades, CREATE_MODEL, CALCULATE_INPUT,
	threshold_abs=THRESHOLD_ABS, threshold_pct=THRESHOLD_PCT, train_vs_test=TRAIN_VS_TEST, num_classes=NUM_CLASSES,
	use_weights_for_classes=USE_WEIGHTS_FOR_CLASSES, use_2_classes_with_random_pick=USE_2_CLASSES_WITH_RANDOM_PICK, 
	num_models=NUM_MODELS, num_epochs=NUM_EPOCHS, verbose=VERBOSE, filter_by_side=-1)
if res is not None:
	data=res['data']
	(all_trades,train_trades,test_trades) = futils.calc_trades_in_every_bin(data, res['num_classes'])
	sys.stderr.write("SHORT: ALL, TRAIN, TEST\n")
	sys.stderr.write("%s, %s, %s\n" % (str(all_trades), str(train_trades), str(test_trades)))
	sys.stderr.write("Testing SHORT:\n")
	test_models( candles, trades, data, res['models'], res['num_classes'], 
		model_type=MODEL_TYPE, filter_by_worst_class=FILTER_BY_WORST_CLASS, print_header=False, 
		prediction_threshold=PREDICTION_THRESHOLD, side=-1)
else:
	sys.stderr.write('Not enough data!\n')
