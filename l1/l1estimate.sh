#!/bin/bash

echo adam_1 
python3 l1estimate.py FILE_NAME=\'test_adam_1.txt\' 
python3 l1estimate.py FILE_NAME=\'test_adam_1.txt\' THRESHOLD=0.2 
echo adam_01 
python3 l1estimate.py FILE_NAME=\'test_adam_01.txt\' 
python3 l1estimate.py FILE_NAME=\'test_adam_01.txt\' THRESHOLD=0.2
echo adam_001 
python3 l1estimate.py FILE_NAME=\'test_adam_001.txt\' 
python3 l1estimate.py FILE_NAME=\'test_adam_001.txt\' THRESHOLD=0.2
echo adam_0001 
python3 l1estimate.py FILE_NAME=\'test_adam_0001.txt\' 
python3 l1estimate.py FILE_NAME=\'test_adam_0001.txt\' THRESHOLD=0.2

echo sgd_01 
python3 l1estimate.py FILE_NAME=\'test_sgd_01.txt\' 
python3 l1estimate.py FILE_NAME=\'test_sgd_01.txt\' THRESHOLD=0.2
echo sgd_001
python3 l1estimate.py FILE_NAME=\'test_sgd_001.txt\' 
python3 l1estimate.py FILE_NAME=\'test_sgd_001.txt\' THRESHOLD=0.2
echo sgd_0001
python3 l1estimate.py FILE_NAME=\'test_sgd_0001.txt\' 
python3 l1estimate.py FILE_NAME=\'test_sgd_0001.txt\' THRESHOLD=0.2
echo rmsprop_001
python3 l1estimate.py FILE_NAME=\'test_rmsprop_001.txt\' 
python3 l1estimate.py FILE_NAME=\'test_rmsprop_001.txt\' THRESHOLD=0.2
echo rmsprop_0001
python3 l1estimate.py FILE_NAME=\'test_rmsprop_0001.txt\' 
python3 l1estimate.py FILE_NAME=\'test_rmsprop_0001.txt\' THRESHOLD=0.2
echo adagrad_01
python3 l1estimate.py FILE_NAME=\'test_adagrad_01.txt\' 
python3 l1estimate.py FILE_NAME=\'test_adagrad_01.txt\' THRESHOLD=0.2
echo adagrad_001
python3 l1estimate.py FILE_NAME=\'test_adagrad_001.txt\' 
python3 l1estimate.py FILE_NAME=\'test_adagrad_001.txt\' THRESHOLD=0.2