import sys
if len(sys.argv) < 2:
	sys.stderr.write( "Use: r.py <a-file-with-signals.csv>" )
	sys.exit()

from datetime import datetime, timedelta
import re, numpy as np, matplotlib.pyplot as plt 
import futils, fmodels
from l2utils import load_and_prepare_data

LOOK_AHEAD = 5
CLASS_THRESHOLD = 1.0
NUM_MODELS = 1
NUM_EPOCHS = 1000 # The number of EPOCHS to train your model through.
TIMEFRAME = 15 # The time frame.
TICKER = "btcusd" # The ticker.
NUM_CLASSES = 3
NUM_MODELS = 1
CREATE_MODEL = fmodels.create_nn_32x16x8  # fmodels.create_nn
CALCULATE_INPUT = fmodels.calculate_input_r # fmodels.calculate_input_i
PRINT_HEADER = False
VERBOSE = True
PREDICTION_THRESHOLD = 1.00001
COMMISSION_PCT=0.0
COMMISSION_ABS=0.0

if re.search( r'\.lst$', sys.argv[1] ): # If a source file passed through command line ends with ".lst"...
	sys.stderr.write('Reading list of source files...\n')
	src = futils.read_list_of_files_with_trades( sys.argv[1] )
	if len(src) == 0:
		sys.stderr.write('Failed to load list of source files. Exiting...\n')
		sys.exit(0)
else: # A source passed though command line is a file name 
	sys.stderr.write('Source file must end with ".lst". Exiting...\n')
	sys.exit(0)

command_line_arguments = ''
for a in range(2, len(sys.argv)): # Reading additional parameters
	m = re.match(r'([a-zA-Z0-9\_ ]+=[0-9a-zA-Z \.\'\"\_]+)', sys.argv[a])
	if m:
		exec(m.group(1))
	command_line_arguments += " " + sys.argv[a]

train = load_and_prepare_data(src, TICKER, TIMEFRAME, lookahead_candles=LOOK_AHEAD, 
	class_threshold=CLASS_THRESHOLD, num_classes=NUM_CLASSES)
if train is None:
	sys.stderr.write('Failed to load or prepare train data. Exiting...\n')
	sys.exit(0)

test = load_and_prepare_data(src, TICKER, TIMEFRAME, lookahead_candles=LOOK_AHEAD, 
	class_threshold=CLASS_THRESHOLD, num_classes=NUM_CLASSES, train_or_test=2)
if test is None:
	sys.stderr.write('Failed to load or prepare test data. Exiting...\n')
	sys.exit(0)

num_features = len(train['inputs'][0])
if len(test['inputs'][0]) != num_features:
	sys.stderr.write('The number of signal sources (features) dirrefs in train and in test. Exiting...\n')
	sys.exit(0)

models = []
num_samples_by_classes = train['num_samples_by_classes']
sys.stderr.write('num_samples_by_classes=%s\n' % (str(num_samples_by_classes)))
num_samples = sum(num_samples_by_classes)
if NUM_CLASSES == 3:
	cw = { 0: (num_samples - num_samples_by_classes[0]) / num_samples, 
		1: (num_samples - num_samples_by_classes[1]) / num_samples,
		2: (num_samples - num_samples_by_classes[2]) / num_samples }
else:
	cw = None
for m in range(NUM_MODELS):
	model = CREATE_MODEL(num_features, NUM_CLASSES) # Creating a model
	model.fit( train['inputs'], train['outputs'], class_weight=cw, epochs=NUM_EPOCHS, verbose=VERBOSE ) # Training the model
	models.append(model)

for t in range( len(test['inputs']) ):
	inp = np.array( [ test['inputs'][t] ] )

	for m in range(NUM_MODELS): # Iterating through all the models trained to obtain predictions
		prediction = models[m].predict_proba( inp )[0]
		sys.stderr.write(str(inp) + '\n')
		sys.stderr.write(str(prediction) + '\n')
		sys.stderr.write(str(test['outputs'][t]) + '\n')
	
