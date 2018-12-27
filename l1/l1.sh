#!/bin/bash

#python3 l1.py OPTIMIZER=keras.optimizers.Adamax(lr=0.002,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0)  
echo adam_0001 
python3 l1.py 'OPTIMIZER=keras.optimizers.Adam(lr=0.0001)' TRAIN_OUT_FILE=\'train_adam_0001.txt\' TEST_OUT_FILE=\'test_adam_0001.txt\' NUM_EPOCHS=3000 
echo adam_001 
python3 l1.py 'OPTIMIZER=keras.optimizers.Adam(lr=0.001)' TRAIN_OUT_FILE=\'train_adam_001.txt\' TEST_OUT_FILE=\'test_adam_001.txt\' NUM_EPOCHS=2000 
echo adam_01 
python3 l1.py 'OPTIMIZER=keras.optimizers.Adam(lr=0.01)' TRAIN_OUT_FILE=\'train_adam_01.txt\' TEST_OUT_FILE=\'test_adam_01.txt\' 
echo adam_1 
python3 l1.py 'OPTIMIZER=keras.optimizers.Adam(lr=0.1)' TRAIN_OUT_FILE=\'train_adam_1.txt\' TEST_OUT_FILE=\'test_adam_1.txt\' 

echo sgd_01 
python3 l1.py 'OPTIMIZER=keras.optimizers.SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)' TRAIN_OUT_FILE=\'train_sgd_01.txt\' TEST_OUT_FILE=\'test_sgd_01.txt\' 
echo sgd_001 
python3 l1.py 'OPTIMIZER=keras.optimizers.SGD(lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)' TRAIN_OUT_FILE=\'train_sgd_001.txt\' TEST_OUT_FILE=\'test_sgd_001.txt\' NUM_EPOCHS=2000 
echo sgd_0001 
python3 l1.py 'OPTIMIZER=keras.optimizers.SGD(lr=0.0001,decay=1e-6,momentum=0.9,nesterov=True)' TRAIN_OUT_FILE=\'train_sgd_0001.txt\' TEST_OUT_FILE=\'test_sgd_0001.txt\' NUM_EPOCHS=3000 
echo rmsprop_001 
python3 l1.py 'OPTIMIZER=keras.optimizers.RMSprop(lr=0.001,rho=0.9,epsilon=None,decay=0.0)' TRAIN_OUT_FILE=\'train_rmsprop_001.txt\' TEST_OUT_FILE=\'test_rmsprop_001.txt\' NUM_EPOCHS=2000 
echo rmsprop_0001 
python3 l1.py 'OPTIMIZER=keras.optimizers.RMSprop(lr=0.0001,rho=0.9,epsilon=None,decay=0.0)' TRAIN_OUT_FILE=\'train_rmsprop_0001.txt\' TEST_OUT_FILE=\'test_rmsprop_0001.txt\' NUM_EPOCHS=3000 
echo adagrad_01
python3 l1.py 'OPTIMIZER=keras.optimizers.Adagrad(lr=0.01,epsilon=None,decay=0.0)' TRAIN_OUT_FILE=\'train_adagrad_01.txt\' TEST_OUT_FILE=\'test_adagrad_01.txt\' 
echo adagrad_001
python3 l1.py 'OPTIMIZER=keras.optimizers.Adagrad(lr=0.001,epsilon=None,decay=0.0)' TRAIN_OUT_FILE=\'train_adagrad_001.txt\' TEST_OUT_FILE=\'test_adagrad_001.txt\'  NUM_EPOCHS=2000 

