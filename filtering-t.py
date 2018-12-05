import sys
if len(sys.argv) < 2:
    print( "Use: filtering <a-file-with-trades.csv>" )
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

# 17-5-2-200-120-10-0.01-0.85-3
THRESHOLD_ABS = 50
THRESHOLD_PCT = 0.0
THRESHOLD_TRADES_NUMBER = 50

PERIOD_OF_INDICATORS = 10
LOOKBACK = 4
LOOKBACK_CANDLES = LOOKBACK + PERIOD_OF_INDICATORS
NUM_FEATURES = 20 # The number of features. This depends on how you implement the "calculate_input()" function. 
#LOOKBACK_CANDLES = 15
#NUM_FEATURES = 5
NUM_CLASSES = 2 # The number of classes depends of what the "calculate_output()" fuction returns (see below)
EPOCHS = 500 # The number of EPOCHS to train your model through.
TIMEFRAME = 30 # The time frame.
TICKER = "btcusd" # The ticker.
TRAIN_VS_TEST = 0.75
start_test_date = None
end_test_date = None

trades = [ 
    {'time':20180501001500001, 'price':9206, 'volume':1, 'positionid':1, 'side':'SELL' },
    { 'time':20180501002515001, 'price':9115.93, 'volume':1, 'positionid':1, 'side':'BUY' },
    { 'time':20180501003000001, 'price':9123.6, 'volume':1, 'positionid':2, 'side':'SELL' },
    { 'time':20180501004115001, 'price':9034.35, 'volume':1, 'positionid':2, 'side':'BUY' }
] 

data = {}

num_models = 0
models = []

scaler = None

def load_data():
    global trades, data, scaler

    data['train_inputs'] = []
    data['train_outputs'] = []
    data['train_trade_num'] = []
    data['test_inputs'] = []
    data['test_outputs'] = []
    data['test_trade_num'] = []    

    trades = utils.read_trades(sys.argv[1])
    if len(trades) == 0:
        return False

    trades = utils.trim_trades( trades )
    if len(trades) == 0:
        return False

    # Searching for min & max... 
    dt_min = trades[0]['time_dt']
    dt_max = trades[0]['closing_time_dt']
    for i in range( 1, len(trades) ):
        if trades[i]['time_dt'] < dt_min:
            dt_min = trades[i]['time_dt']
        if trades[i]['closing_time_dt'] > dt_max:
            dt_max = trades[i]['closing_time_dt']

    print(dt_min - timedelta(minutes = TIMEFRAME*LOOKBACK_CANDLES))
    print(dt_max)

    num_days = (dt_max - dt_min).days
    num_days_in_test = int(num_days * TRAIN_VS_TEST)

    dt_train_end = dt_min + timedelta(num_days_in_test)
    print(dt_train_end)

    candles = import_candles( TICKER, TIMEFRAME, dt_min - timedelta(minutes = TIMEFRAME*LOOKBACK_CANDLES), dt_max )

    num_bin_1 = 0
    num_bin_0 = 0

    for t in range(len(trades)):
        for c in range(len(candles)-LOOKBACK_CANDLES-1,-1,-1):
            if candles['time'][c] >= trades[t]['time']:
                trades[t]['start_candle'] = c
                inp = calculate_input(candles[c+1:c+LOOKBACK_CANDLES+1].reset_index())

                profit = trades[t]['profit']
                profit_pct = (trades[t]['profit'] * 100.0) / trades[t]['price']
                if profit > THRESHOLD_ABS and profit_pct > THRESHOLD_PCT:
                    output = [0, 1]
                    num_bin_1 += 1
                else:
                    output = [1, 0]
                    num_bin_0 += 1

                if trades[t]['time_dt'] < dt_train_end:
                    data['train_inputs'].append(inp)
                    data['train_outputs'].append(output)
                    data['train_trade_num'].append(t)
                else:
                    data['test_inputs'].append(inp)
                    data['test_outputs'].append(output)
                    data['test_trade_num'].append(t)
                break
    # end of "for"
    data['num_bin_1'] = num_bin_1
    data['num_bin_0'] = num_bin_0

    data['train_inputs'] = np.array(data['train_inputs'])
    data['train_outputs'] = np.array(data['train_outputs'])
    data['test_inputs'] = np.array(data['test_inputs'])
    data['test_outputs'] = np.array(data['test_outputs'])

    scaler = StandardScaler() # Creating an instance of a scaler.
    scaler.fit(data['train_inputs']) # Fitting the scaler.
    data['train_inputs'] = scaler.transform(data['train_inputs']) # Normalizing data

    return True
# end of load_data


def load_data_and_trainmodels():
    global num_models, models

    if not load_data():
        print("Can't load data! Exiting...")
        sys.exit();

    num_models = int( float(data['num_bin_0']) / float(data['num_bin_1']) + 0.9 )
    for m in range(num_models):
        if num_models > 1:
            inputs = np.array(data['train_inputs'])
            outputs = np.array(data['train_outputs'])
            len_outputs = np.shape(data['train_outputs'])[0]
            for i in range(data['num_bin_1']):
                # random pick
                while(True):
                    random_pick = randint(0, len_outputs-1 )
                    if data['train_outputs'][random_pick][1] == 1:
                        continue

                    inputs = np.delete(inputs, (random_pick), axis=0)
                    outputs = np.delete(outputs, (random_pick), axis=0)
                    len_outputs -= 1
                    break
        else:
            inputs = data['train_inputs']
            outputs = data['train_outputs']
        #print(inputs)
        #print(outputs)
        model = create_model() # Creating a model
        model.fit( inputs, outputs, epochs=EPOCHS ) # Training the model
        models.append( model )


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
    m.add(Dense(NUM_CLASSES, activation='softmax'))
    # Compiling the model
    m.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return m
# end of create_model()


def calculate_input(candles):
    input_vec = []

    #for i in range(NUM_FEATURES): # Returns
    #    input_vec.append( (candles['close'][0] - candles['open'][i]) / candles['close'][0] )
    #return(input_vec)    

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


load_data_and_trainmodels() 

print( 'N(bin1)=%d, N(bin0)=%d' % (data['num_bin_1'], data['num_bin_0']) )
input("Press <ENTER>")

num_trades = 0
num_trades_oked = 0
filtered_right = 0
filtered_wrong = 0
profit_actual = 0
profit_optimized = 0
for t in range(len(data['test_inputs'])):
    inp = np.array( [ data['test_inputs'][t] ] )
    inp = scaler.transform(inp)        
    trade_num = data['test_trade_num'][t]
    profit = trades[trade_num]['profit']
    num_models_that_allowed_trade = 0

    for m in range(num_models):
        prediction = models[m].predict_proba( inp )
        print(prediction)
        print(data['test_outputs'][t])
        if prediction[0][1] > prediction[0][0] * 1.25:
            num_models_that_allowed_trade += 1
    
    profit_actual += profit
    num_trades += 1

    if num_models_that_allowed_trade < num_models:
        continue

    profit_optimized += profit
    num_trades_oked += 1    
    if data['test_outputs'][t][1] == 1:
        filtered_wrong += 1
    else:
        filtered_right += 1

    print("Trades (actual/oked): %d (%d), filtered right: %d, filtered wrong: %d, profit_actual=%f, profit_optimized=%f" % \
        (num_trades, num_trades_oked, filtered_right, filtered_wrong, profit_actual, profit_optimized))

'''
print( "%f (%d)" % ( pnl, num_positions ) )
plt.plot(pnl)
plt.text( 1, 0, _num_positions )
plt.show()
'''

'''




            outputs = np.empty( shape=(data['num_bin_1'] * 2, 2))
            inputs = np.empty( shape=(data['num_bin_1'] * 2, NUM_FEATURES))
            for i in range(data['num_bin_1']):
                # random pick
                while(True):
                    random_pick = randint(0, data['num_bin_0'] )
                    if data['train_outputs'][random_pick][1] == 1:
                        continue

                    outputs[i][0] = 1 # data['train_outputs'][random_pick][0]
                    outputs[i][1] = 0 # data['train_outputs'][random_pick][1]
                    for f in range(NUM_FEATURES):
                        inputs[i][f] = data['train_inputs'][random_pick][f]
'''
