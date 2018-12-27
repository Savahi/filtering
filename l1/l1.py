import sys
import re
from datetime import datetime
from tradingene.data.load import import_data
from tradingene.algorithm_backtest.tng import TNG
import tradingene.backtest_statistics.backtest_statistics as bs
import keras
import numpy as np
import l1models

NUM_EPOCHS = 1000 # The number of epochs to train your model through.
TIMEFRAME = 15 # The time frame.
TICKER = "btcusd" # The ticker.
START_TRAIN_DATE = datetime(2018, 5, 1) # When a train period starts...
END_TRAIN_DATE = datetime(2018, 6, 1) # When the train period ends and the test starts...
END_TEST_DATE = datetime(2018, 6, 6) # When the test ends...

CREATE_MODEL_FN = l1models.create_nn
OPTIMIZER = None
ACTIVATION = None
CALC_INPUT_FN = l1models.calculate_input_sum
CALC_OUTPUT_FN = l1models.calculate_output_mean

TEST_OUT_FILE = 'test.txt'
TRAIN_OUT_FILE = 'train.txt'

VERBOSE = False

lookback = 0
num_features = 0
lookforward = 0
num_classes = 0

model = None # To store a model to be trained and serve as a "price-predicting tool"
file_handle = None

command_line_arguments = ''
for a in range(1, len(sys.argv)): # Reading additional parameters
    m = re.match(r'([a-zA-Z0-9\_ ]+=[0-9a-zA-Z \.\'\"\_\)\(\,\-\=)]+)', sys.argv[a])
    if m:
        exec(m.group(1))
    command_line_arguments += ' ' + sys.argv[a]


def run( start_date, end_date, file_name ):
    global file_handle, model

    try:
        file_handle = open(file_name, "w")

        if model is None:
            model = prepare_model() # Creating an ML-model.
        alg = TNG(start_date, end_date) # Creating an instance of environment to run algorithm in.
        alg.addInstrument(TICKER) # Adding an instrument.
        alg.addTimeframe(TICKER, TIMEFRAME) # Adding a time frame. 
        alg.run_backtest(on_bar) # Backtesting...
        del alg

    except IOError:
        sys.stderr.write( "Error while running a backtest for %s.\n" % (file_name) )
    else:
        sys.stderr.write( "File %s done.\n" % (file_name) )

    if file_handle is not None:
        file_handle.close()
        file_handle = None

    return model
# end of def 



def prepare_model():
    global lookback, num_features, lookforward, num_classes

    lookback, num_features = CALC_INPUT_FN('query_lookback')
    lookforward, num_classes = CALC_OUTPUT_FN('query_lookforward')

    l1models.calculate_thresholds_for_equal_class_sets( TICKER, TIMEFRAME, START_TRAIN_DATE, END_TRAIN_DATE, CALC_OUTPUT_FN )

    data = import_data(
        TICKER, TIMEFRAME, START_TRAIN_DATE, END_TRAIN_DATE, 
        CALC_INPUT_FN, lookback, CALC_OUTPUT_FN, lookforward,
        split = (100, 0, 0) # This time we need only a train set (100% for train set, 0% for test and validation ones)
    )

    model = CREATE_MODEL_FN( num_features, num_classes, optimizer=OPTIMIZER )
    one_hot_train_outputs = keras.utils.to_categorical(data['train_output'], num_classes=num_classes) # Performing one-hot encoding

    # Calculating class weights
    num_samples = len(data['train_output'])
    num_samples_by_classes = [0]*num_classes
    for n in range(num_samples):
        index = int(data['train_output'][n][0])
        num_samples_by_classes[index] += 1
    print('NUM. SAMPLES BY CLASSES:')
    print(str(num_samples_by_classes))
    cw = {}
    for c in range(num_classes):
        cw[c] = (num_samples - num_samples_by_classes[c]) / num_samples 
    sys.stderr.write('Fitting...\n')
    model.fit(data['train_input'], one_hot_train_outputs, class_weight=cw, epochs=NUM_EPOCHS, verbose=VERBOSE) # Training the model

    sys.stderr.write('Evaluating train...\n')
    scores = model.evaluate( data['train_input'], one_hot_train_outputs, verbose=VERBOSE)
    metrics = "%s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100)
    sys.stderr.write( metrics )    

    sys.stderr.write('Evaluating test...\n')
    data = import_data(
        TICKER, TIMEFRAME, END_TRAIN_DATE, END_TEST_DATE, 
        CALC_INPUT_FN, lookback, CALC_OUTPUT_FN, lookforward,
        split = (0, 0, 100) # This time we need only a test set (100% for train set, 0% for test and validation ones)
    )
    one_hot_test_outputs = keras.utils.to_categorical(data['test_output'], num_classes=num_classes) # Performing one-hot encoding
    scores = model.evaluate( data['test_input'], one_hot_test_outputs, verbose=VERBOSE)
    metrics = "%s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100)
    sys.stderr.write( metrics )    

    return model
# end of prepare_model


def on_bar(instrument):
    inp = CALC_INPUT_FN(instrument.rates[1:lookback+1])

    probabilities = model.predict_proba([inp])[0]
    str_probabilities = ''
    for p in range(len(probabilities)):
        if p > 0:
            str_probabilities += ';'
        str_prob = str( probabilities[p] )
        if str_prob.find('.') == -1:
            str_prob += '.0'
        str_probabilities += str_prob

    # Calculating output for lookforward candle bars back
    rates_for_calc_outp = instrument.rates[1:lookforward+1]
    rates_for_calc_outp = rates_for_calc_outp[::-1]
    outp = CALC_OUTPUT_FN(rates_for_calc_outp)[0]
    # index, sell, buy, open, profit, pos_prof/outp, time, pos
    time_str = str(int(instrument.rates[0]['time']))
    output = '0,0,0,%f,0,%d,%s,%s\n' % (instrument.rates[0]['open'], int(outp), time_str, str_probabilities)
    # print(output)
    file_handle.write( output )
# end of onBar()

run( START_TRAIN_DATE, END_TRAIN_DATE, TRAIN_OUT_FILE )
run( END_TRAIN_DATE, END_TEST_DATE, TEST_OUT_FILE )
