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
import keras
import numpy as np
import utils
import matplotlib.pyplot as plt 

# 17-5-2-200-120-10-0.01-0.85-3
LOOKBACK = 7
NUM_FEATURES = LOOKBACK-1 # The number of features. This depends on how you implement the "calculate_input()" function. 
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

NUM_MODELS = 3
models = []
for i in range(NUM_MODELS):
    models.append(None)

def load_data():
    global trades, data

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

    print(dt_min - timedelta(minutes=TIMEFRAME*LOOKBACK))
    print(dt_max)

    num_days = (dt_max - dt_min).days
    num_days_in_test = int(num_days * TRAIN_VS_TEST)

    dt_train_end = dt_min + timedelta(num_days_in_test)
    print(dt_train_end)

    candles = import_candles( TICKER, TIMEFRAME, dt_min - timedelta(minutes=TIMEFRAME*LOOKBACK), dt_max )

    for t in range(len(trades)):
        for c in range(len(candles)-LOOKBACK-1,-1,-1):
            if candles['time'][c] >= trades[t]['time']:
                trades[t]['start_candle'] = c
                inp = calculate_input(candles[c+1:c+LOOKBACK+1].reset_index())

                profit = trades[t]['profit']
                if profit > 0:
                    output = [0, 1]
                else:
                    output = [1, 0]

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
    data['train_inputs'] = np.array(data['train_inputs'])
    data['train_outputs'] = np.array(data['train_outputs'])
    data['test_inputs'] = np.array(data['test_inputs'])
    data['test_outputs'] = np.array(data['test_outputs'])
    return True
# end of load_data


def load_data_and_trainmodels():
    if not load_data():
        print("Can't load data! Exiting...")
        sys.exit();

    np.shape(data['train_inputs'])
    np.shape(data['train_outputs'])

    for i in range(NUM_MODELS):
        m = create_model() # Creating a model
        m.fit( data['train_inputs'], data['train_outputs'], epochs=EPOCHS ) # Training the model
        models[i] = m


def create_model():
    global _data_rows, NUM_FEATURES, NUM_CLASSES

    m = Sequential()
    # The number of nodes in the first hidden layer equals the number of features
    m.add(Dense(units=NUM_FEATURES*4, activation='tanh', input_dim=NUM_FEATURES, kernel_initializer=he_uniform(1)))
    # Adding another hidden layer 
    m.add(Dense(NUM_FEATURES*4, activation='tanh'))
    # Adding an output layer
    m.add(Dense(NUM_CLASSES, activation='softmax'))
    # Compiling the model
    m.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return m


def calculate_input(candles):
    input_vec = []
    for i in range(NUM_FEATURES):
        input_vec.append( np.log(candles['close'][i+1] / candles['close'][0]) * 100.0 )
    print(input_vec)
    return(input_vec)


load_data_and_trainmodels() 

prediction_balance = 0
prediction_tries = 0
filtered_right = 0
filtered_wrong = 0
for t in range(len(data['test_inputs'])):
    for m in range(NUM_MODELS):
        inp = np.array( [ data['test_inputs'][t] ] )
        prediction = models[m].predict_proba( inp )
        print(prediction)
        print(data['test_outputs'][t])
        if prediction[0][1] > prediction[0][0]:
            if data['test_outputs'][t][1] == 1:
                prediction_balance += 1
            else:
                prediction_balance -= 1
        elif prediction[0][1] < prediction[0][0]:
            if data['test_outputs'][t][1] == 1:
                prediction_balance -= 1
                filtered_wrong += 1
            else:
                prediction_balance += 1
                filtered_right += 1
        prediction_tries += 1
        print("Prediction balance: %d (%d), filtered right: %d, filtered wrong: %d" % \
            (prediction_balance, prediction_tries, filtered_right, filtered_wrong))

'''
print( "%f (%d)" % ( pnl, num_positions ) )
plt.plot(pnl)
plt.text( 1, 0, _num_positions )
plt.show()
'''