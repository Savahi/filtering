from datetime import datetime
from tradingene.data.load import import_data
from tradingene.algorithm_backtest.tng import TNG
import tradingene.backtest_statistics.backtest_statistics as bs
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import he_normal, he_uniform
from keras.layers.normalization import BatchNormalization
import keras
import numpy as np

LOOKBACK = 9 # How many prior candle bars each train sample embraces.
NUM_FEATURES = LOOKBACK # The number of features. This depends on how you implement the "calculate_input()" function. 
    # This time the number of features equals the "lookback" period.   
LOOKFORWARD = 3 # How far in the future the algorithm "looks" and foresees.
NUM_CLASSES = 3 # The number of classes depends of what the "calculate_output()" fuction returns (see below)
NUM_EPOCHS = 100 # The number of epochs to train your model through.
TIMEFRAME = 60 # The time frame.
TICKER = "btcusd" # The ticker.
START_TRAIN_DATE = datetime(2017, 5, 1) # When a train period starts...
END_TRAIN_DATE = datetime(2017, 6, 1) # When the train period ends and the test starts...
END_TEST_DATE = datetime(2017, 6, 6) # When the test ends...

THRESHOLD = 0.001 # A threshold to devide outputs into classes (please refer to the "calculate_output()" function below)

alg = None # An instance of the "TNG" class used for simulated trading

model = None # To store a model to be trained and serve as a "price-predicting tool"

def prepare_model():
    data = import_data(
        TICKER, TIMEFRAME, START_TRAIN_DATE, END_TRAIN_DATE, 
        calculate_input, LOOKBACK, calculate_output, LOOKFORWARD,
        split = (100, 0, 0) # This time we need only a train set (100% for train set, 0% for test and validation ones)
    )

    # Creating a model...
    model = Sequential() # Creating a model class instance
    # The number of nodes in the first hidden layer equals the number of features multiplied by 2
    model.add(Dense(units=NUM_FEATURES*2, activation='tanh', input_dim=NUM_FEATURES, kernel_initializer=he_uniform(1)))
    # Adding another hidden layer 
    model.add(Dense(NUM_FEATURES*2, activation='tanh'))
    # Adding an output layer
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    # Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    one_hot_train_outputs = keras.utils.to_categorical(data['train_output'], num_classes=NUM_CLASSES) # Performing one-hot encoding
    model.fit(data['train_input'], one_hot_train_outputs, epochs=NUM_EPOCHS) # Training the model
    return model
# end of prepare_model


def calculate_input(data):
    input_vec = np.zeros(NUM_FEATURES) # A vector to store inputs 
    for i in range(LOOKBACK):  
        input_vec[i] = (data['open'][i] - data['close'][0]) / data['close'][0] 
    return np.array([input_vec])


def calculate_output(data):
    for i in range(LOOKFORWARD):
        if (data['high'][i] - data['open'][0]) / data['open'][0] > THRESHOLD: # If price breaks the threshold up...
            return [2] # ... it makes class "2"
        elif (data['low'][i] - data['open'][0]) / data['open'][0] < -THRESHOLD: # If price breaks the threshold down...
            return [0] # ... it make class "0"
    return [1] # If the threshold hasn/t been broken neither up nor down - it's class "1"  


def onBar(instrument):
    inp = calculate_input(instrument.rates[1:LOOKBACK+1])

    prediction = _model.predict_classes([inp])[0]
    if prediction == 2: # Class "2" predicts price to surge up
        alg.buy()
        tp = instrument.rates[0]['open'] * THRESHOLD # Setting the take profit as big as the price threshold
        sl = tp                                       # Setting the stop loss equal to the take profit.
        alg.setTP(profit = tp)
        alg.setSL(loss = sl)
    elif prediction == 0: # Class "0" predicts price to fall down
        alg.sell()
        tp = instrument.rates[0]['open'] * THRESHOLD # Setting the take profit as big as the price threshold
        sl = tp                                       # Setting the stop loss equal to the take profit.
        alg.setTP(profit = tp)
        alg.setSL(loss = sl)
# end of onBar()

model = prepare_model() # Creating an ML-model.
alg = TNG(END_TRAIN_DATE, END_TEST_DATE) # Creating an instance of environment to run algorithm in.
alg.addInstrument(TICKER) # Adding an instrument.
alg.addTimeframe(TICKER, TIMEFRAME) # Adding a time frame. 
alg.run_backtest(onBar) # Backtesting...

stat = bs.BacktestStatistics(alg) # Retrieving statistics of the backtest

pnl = stat.calculate_PnL()
num_positions = stat.calculate_number_of_trades()
print("pnl=%f, num_positions=%d" % (pnl, num_positions))

stat.backtest_results() # Displaying the backtest statistics
