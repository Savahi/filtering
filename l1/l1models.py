import numpy as np
from sklearn.svm import SVC
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.initializers import he_normal, he_uniform
from keras.layers.normalization import BatchNormalization
from tradingene.data.load import import_candles

def create_nn_32x16x8( num_features, num_classes=2 ):
    m = Sequential()
    m.add(Dense(units=32, activation='tanh', input_dim=num_features, kernel_initializer=he_uniform(1)))
    m.add(Dense(16, activation='tanh'))
    m.add(Dense(8, activation='tanh'))
    m.add(Dense(num_classes, activation='softmax'))
    # Compiling the model
    m.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return m
# end of create_model()


def create_nn( num_features, num_classes=2, optimizer=None, activation=None ):
    if isinstance(num_features,str):
        return( True ) # Returns "True" if the model requires one-hot encoding and "False" otherwise]

    if optimizer is None:
        optimizer = 'adam'
    if activation is None:
        activation = 'tanh'

    m = Sequential()
    m.add(Dense(units=num_features*4, activation=activation, input_dim=num_features, kernel_initializer=he_uniform(1)))
    m.add(Dense(num_features*4, activation=activation))
    m.add(Dense(num_classes, activation='softmax'))
    # Compiling the model
    #m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    m.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])    
    return m
# end of create_model()

calculate_input_lookback = 25

def calculate_input_num(candles):
    lookback = calculate_input_lookback
    calc_values_at = [ 1, 4, 8, 13, 24, 48 ]
    
    if isinstance(candles,str): # Querying only...
        num_features = 0
        for r in range(len(calc_values_at)):
            if calc_values_at[r] < lookback:
                num_features += 1
        return( lookback, num_features ) # Returning 1) a lookback period involved and 2) the number of features generated

    input_vec = []
    last_price = candles['close'][0]
    num_upper = 0
    num_lower = 0
    for i in range(lookback):
        if not (candles['high'][i] < last_price):
            num_upper += 1
        if not( candles['high'][i] > last_price):
            num_lower += 1
        if not (candles['low'][i] < last_price):
            num_upper += 1
        if not( candles['low'][i] > last_price):
            num_lower += 1
        if not (candles['open'][i] < last_price):
            num_upper += 1
        if not( candles['open'][i] > last_price):
            num_lower += 1

        for r in range(len(calc_values_at)):
            if i == calc_values_at[r]:
                upper_plus_lower = num_upper + num_lower
                if upper_plus_lower > 0:
                    input_vec.append( 100.0 * float(num_upper) / float(upper_plus_lower) )
                else:
                    input_vec.append( 100.0 * 0.5 )

    return([input_vec])   
# end of ...


def calculate_input_sum(candles):
    lookback = calculate_input_lookback
    calc_values_at = [ 1, 4, 8, 13, 24, 48 ]
    
    if isinstance(candles,str): # Querying only...
        num_features = 0
        for r in range(len(calc_values_at)):
            if calc_values_at[r] < lookback:
                num_features += 1
        return( lookback, num_features ) # Returning 1) a lookback period involved and 2) the number of features generated

    input_vec = []
    last_price = candles['close'][0]
    sum_upper = 0
    sum_lower = 0
    for i in range(lookback):
        if not (candles['high'][i] < last_price):
            sum_upper += candles['high'][i] - last_price
        if not( candles['high'][i] > last_price):
            sum_lower += last_price - candles['high'][i]
        if not (candles['low'][i] < last_price):
            sum_upper += candles['low'][i] - last_price
        if not( candles['low'][i] > last_price):
            sum_lower += last_price - candles['low'][i]
        if not (candles['open'][i] < last_price):
            sum_upper += candles['open'][i] - last_price
        if not( candles['open'][i] > last_price):
            sum_lower += last_price - candles['open'][i]

        for r in range(len(calc_values_at)):
            if i == calc_values_at[r]:
                upper_plus_lower = float(sum_upper + sum_lower)
                if upper_plus_lower > 0.0:
                    input_vec.append( 100.0 * float(sum_upper) / upper_plus_lower )
                else:
                    input_vec.append( 100.0 * 0.5 )
                break
    return([input_vec])   
# end of ...


calculate_output_settings = { 'lookforward':5, 'threshold0':0.998, 'threshold1':1.002 }

def calculate_output_mean(candles, return_raw_value=False):
    lookforward = calculate_output_settings['lookforward']
    threshold0 = calculate_output_settings['threshold0']
    threshold1 = calculate_output_settings['threshold1']

    if isinstance(candles,str):
        return( lookforward, 3 ) # Returning 1) the lookforward period and 2) the number of classes

    meanh = np.mean(candles['high'][:lookforward])
    meanl = np.mean(candles['low'][:lookforward])
    meanc = np.mean(candles['close'][:lookforward])
    mean = (meanh + meanl + meanc)/3.0

    relative_mean = mean / candles['open'][0]

    if return_raw_value:
        return relative_mean

    if relative_mean > threshold1: # If price breaks the threshold up...
        return [2] # ... it makes class "2"
    elif relative_mean < threshold0: # If price breaks the threshold down...
        return [0] # ... it make class "0"
    return [1] # If the threshold hasn/t been broken neither up nor down - it's class "1"  
# end of def


def calculate_output_mean2(candles, return_raw_value=False):
    lookforward = calculate_output_settings['lookforward']
    threshold0 = calculate_output_settings['threshold0']
    threshold1 = calculate_output_settings['threshold1']

    if isinstance(candles,str):
        return( lookforward, 2 ) # Returning 1) the lookforward period and 2) the number of classes

    meanh = np.mean(candles['high'][:lookforward])
    meanl = np.mean(candles['low'][:lookforward])
    meanc = np.mean(candles['close'][:lookforward])
    mean = (meanh + meanl + meanc)/3.0

    relative_mean = mean / candles['open'][0]
    if relative_mean < 1.0:
        relative_mean = 2.0 - relative_mean;

    if return_raw_value:
        return relative_mean

    if relative_mean > threshold0: # If price breaks the threshold up...
        return [1] # ... it makes class "2"
    return [0] # If the threshold hasn/t been broken neither up nor down - it's class "1"  
# end of def


def calculate_thresholds_for_equal_class_sets( ticker, timeframe, start_date, end_date, calculate_output_function ):
    lookforward, num_classes = calculate_output_function('query_lookforward')

    candles = import_candles(ticker, timeframe, start_date, end_date)
    len_candles = len(candles)

    # loading candles... 
    raw_values = []
    for p in range( lookforward, len_candles+1 ):
        c = candles.iloc[ p-lookforward : p ]
        c = c.iloc[::-1]
        c = c.reset_index()
        v = calculate_output_function(c,True)
        raw_values.append(v)
    
    raw_values = np.sort(raw_values)
    print(raw_values)
    # sorting...
    len_raw_values = len(raw_values)
    for c in range(num_classes-1):
        index = int( float(len_raw_values) * float(c+1) / float(num_classes) + 0.5 )
        calculate_output_settings['threshold'+str(c)] = raw_values[index]        
    print(calculate_output_settings)



