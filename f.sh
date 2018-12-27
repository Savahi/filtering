#!/bin/bash

#python3 f.py f.lst SRC_FORMAT=\"nn\" TIMEFRAME=15 TRAIN_VS_TEST=None NUM_CLASSES=None TICKER=\"btcusd\" FILTER_BY_WORST_CLASS=True THRESHOLD_PCT=0.7 PREDICTION_THRESHOLD=1.25 >> f.txt
#python3 f.py f.lst SRC_FORMAT=\"nn\" TIMEFRAME=15 TRAIN_VS_TEST=None NUM_CLASSES=3 FILTER_BY_WORST_CLASS=True TICKER=\"btcusd\" THRESHOLD_PCT=0.7 PREDICTION_THRESHOLD=1.25 >> f.txt
#python3 f.py f.lst SRC_FORMAT=\"nn\" TIMEFRAME=15 TRAIN_VS_TEST=None NUM_CLASSES=5 FILTER_BY_WORST_CLASS=True TICKER=\"btcusd\" THRESHOLD_PCT=0.7 PREDICTION_THRESHOLD=1.25 >> f.txt

python3 f.py f2.lst NUM_CLASSES=3 SAMPLES_MULTIPLIER=0.05 NUM_EPOCHS=2000 THRESHOLD_PCT=0.7 PREDICTION_THRESHOLD=0.05
python3 f.py f2.lst NUM_CLASSES=3 SAMPLES_MULTIPLIER=0.1 NUM_EPOCHS=2000 THRESHOLD_PCT=0.7 PREDICTION_THRESHOLD=0.05
python3 f.py f2.lst NUM_CLASSES=3 SAMPLES_MULTIPLIER=0.15 NUM_EPOCHS=2000 THRESHOLD_PCT=0.7 PREDICTION_THRESHOLD=0.05
python3 f.py f2.lst NUM_CLASSES=3 'OPTIMIZER=keras.optimizers.Adam(lr=0.0001)' SAMPLES_MULTIPLIER=0.05 NUM_EPOCHS=3000 THRESHOLD_PCT=0.7 PREDICTION_THRESHOLD=0.1
python3 f.py f2.lst NUM_CLASSES=3 'OPTIMIZER=keras.optimizers.Adam(lr=0.0001)' SAMPLES_MULTIPLIER=0.1 NUM_EPOCHS=3000 THRESHOLD_PCT=0.7 PREDICTION_THRESHOLD=0.1
python3 f.py f2.lst NUM_CLASSES=3 'OPTIMIZER=keras.optimizers.Adam(lr=0.0001)' SAMPLES_MULTIPLIER=0.15 NUM_EPOCHS=3000 THRESHOLD_PCT=0.7 PREDICTION_THRESHOLD=0.1

#python3 f.py f2.lst SRC_FORMAT=\"nn\" TIMEFRAME=15 TRAIN_VS_TEST=None NUM_CLASSES=5 TICKER=\"btcusd\"  SAMPLES_MULTIPLIER=0.05 VERBOSE=True THRESHOLD_PCT=0.7 PREDICTION_THRESHOLD=1.01
#python3 f.py f2.lst SRC_FORMAT=\"nn\" TIMEFRAME=15 TRAIN_VS_TEST=None NUM_CLASSES=None TICKER=\"btcusd\"  SAMPLES_MULTIPLIER=0.05 VERBOSE=True THRESHOLD_PCT=0.7 PREDICTION_THRESHOLD=1.01

: '
python3 f.py f/266.lst SRC_FORMAT=\"nn\" TIMEFRAME=15 TRAIN_VS_TEST=None NUM_CLASSES=None TICKER=\"btcusd\" THRESHOLD_PCT=0.7 >> f.txt
python3 f.py f/288.lst SRC_FORMAT=\"nn\" TIMEFRAME=15 TRAIN_VS_TEST=None NUM_CLASSES=None TICKER=\"btcusd\" THRESHOLD_PCT=0.7 >> f.txt
python3 f.py f/330.lst SRC_FORMAT=\"nn\" TIMEFRAME=15 TRAIN_VS_TEST=None NUM_CLASSES=None TICKER=\"btcusd\" THRESHOLD_PCT=0.7 >> f.txt
python3 f.py f/362.lst SRC_FORMAT=\"nn\" TIMEFRAME=15 TRAIN_VS_TEST=None NUM_CLASSES=None TICKER=\"btcusd\" THRESHOLD_PCT=0.7 >> f.txt
python3 f.py f/369.lst SRC_FORMAT=\"nn\" TIMEFRAME=15 TRAIN_VS_TEST=None NUM_CLASSES=None TICKER=\"btcusd\" THRESHOLD_PCT=0.7 >> f.txt
python3 f.py f/491.lst SRC_FORMAT=\"nn\" TIMEFRAME=15 TRAIN_VS_TEST=None NUM_CLASSES=None TICKER=\"btcusd\" THRESHOLD_PCT=0.7 >> f.txt
python3 f.py f/497.lst SRC_FORMAT=\"nn\" TIMEFRAME=15 TRAIN_VS_TEST=None NUM_CLASSES=None TICKER=\"btcusd\" THRESHOLD_PCT=0.7 >> f.txt
'

: '
python3 f.py data/ivan/288.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_r >> f.txt
python3 f.py data/ivan/362.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt

python3 f.py data/ivan/369.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt
python3 f.py data/ivan/362.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt

python3 f.py data/abgar/si/1.csv TIMEFRAME=60 TICKER=\"si\" PRINT_HEADER=True >> f.txt
python3 f.py data/abgar/si/2.csv TIMEFRAME=15 TICKER=\"si\" >> f.txt
python3 f.py data/abgar/si/3.csv TIMEFRAME=15 TICKER=\"si\" >> f.txt
python3 f.py data/abgar/si/4.csv TIMEFRAME=15 TICKER=\"si\" >> f.txt

python3 f.py data/abgar/si/1.csv TIMEFRAME=60 TICKER=\"si\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/si/2.csv TIMEFRAME=15 TICKER=\"si\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/si/3.csv TIMEFRAME=15 TICKER=\"si\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/si/4.csv TIMEFRAME=15 TICKER=\"si\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt

python3 f.py data/abgar/rts/1.csv TIMEFRAME=60 TICKER=\"rts\" PRINT_HEADER=True >> f.txt
python3 f.py data/abgar/rts/2.csv TIMEFRAME=15 TICKER=\"rts\" >> f.txt
python3 f.py data/abgar/rts/3.csv TIMEFRAME=15 TICKER=\"rts\" >> f.txt
python3 f.py data/abgar/rts/4.csv TIMEFRAME=15 TICKER=\"rts\" >> f.txt

python3 f.py data/abgar/rts/1.csv TIMEFRAME=60 TICKER=\"rts\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/rts/2.csv TIMEFRAME=15 TICKER=\"rts\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/rts/3.csv TIMEFRAME=15 TICKER=\"rts\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/rts/4.csv TIMEFRAME=15 TICKER=\"rts\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt

python3 f.py data/abgar/btc/1.csv TIMEFRAME=60 TICKER=\"btcusd\" PRINT_HEADER=True >> f.txt
python3 f.py data/abgar/btc/2.csv TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt
python3 f.py data/abgar/btc/3.csv TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt
python3 f.py data/abgar/btc/4.csv TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt

python3 f.py data/abgar/btc/1.csv TIMEFRAME=60 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/btc/2.csv TIMEFRAME=15 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/btc/3.csv TIMEFRAME=15 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/btc/4.csv TIMEFRAME=15 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt

python3 f.py data/ivan/tr_decis_n_0_e_288.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt
python3 f.py data/ivan/tr_decis_n_0_e_330.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt
python3 f.py data/ivan/tr_decis_n_0_e_369.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt
python3 f.py data/ivan/tr_decis_n_1_e_362.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt
python3 f.py data/ivan/tr_decis_n_1_e_491.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt
python3 f.py data/ivan/tr_decis_n_1_e_497.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt
python3 f.py data/ivan/tr_decis_n_2_e_266.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt

python3 f.py data/ivan/tr_decis_n_0_e_288.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_0_e_330.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_0_e_369.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_1_e_362.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_1_e_491.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_1_e_497.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_2_e_266.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt

python3 f.py data/ivan/tr_decis_n_2_e_346.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt
python3 f.py data/ivan/tr_decis_n_2_e_393.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt
python3 f.py data/ivan/tr_decis_n_3_e_390.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt
python3 f.py data/ivan/tr_decis_n_3_e_446.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt
python3 f.py data/ivan/tr_decis_n_3_e_487.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" >> f.txt

python3 f.py data/ivan/tr_decis_n_2_e_346.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_2_e_393.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_3_e_390.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_3_e_446.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_i CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_3_e_487.csv SRC_FORMAT=\"nn\" TIMEFRAME=15 TICKER=\"btcusd\" CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
'