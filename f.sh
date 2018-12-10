#!/bin/bash

python3 f.py data/abgar/si/1.csv TIMEFRAME=60 TICKER=\'si\' PRINT_HEADER=True >> f.txt
python3 f.py data/abgar/si/2.csv TIMEFRAME=15 TICKER=\'si\' >> f.txt
python3 f.py data/abgar/si/3.csv TIMEFRAME=15 TICKER=\'si\' >> f.txt
python3 f.py data/abgar/si/4.csv TIMEFRAME=15 TICKER=\'si\' >> f.txt

python3 f.py data/abgar/si/1.csv TIMEFRAME=60 TICKER=\'si\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/si/2.csv TIMEFRAME=15 TICKER=\'si\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/si/3.csv TIMEFRAME=15 TICKER=\'si\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/si/4.csv TIMEFRAME=15 TICKER=\'si\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt


"""
python3 f.py data/abgar/rts/1.csv TIMEFRAME=60 TICKER=\'rts\' PRINT_HEADER=True >> f.txt
python3 f.py data/abgar/rts/2.csv TIMEFRAME=15 TICKER=\'rts\' >> f.txt
python3 f.py data/abgar/rts/3.csv TIMEFRAME=15 TICKER=\'rts\' >> f.txt
python3 f.py data/abgar/rts/4.csv TIMEFRAME=15 TICKER=\'rts\' >> f.txt

python3 f.py data/abgar/rts/1.csv TIMEFRAME=60 TICKER=\'rts\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/rts/2.csv TIMEFRAME=15 TICKER=\'rts\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/rts/3.csv TIMEFRAME=15 TICKER=\'rts\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/rts/4.csv TIMEFRAME=15 TICKER=\'rts\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt

python3 f.py data/abgar/btc/1.csv TIMEFRAME=60 TICKER=\'btcusd\' PRINT_HEADER=True >> f.txt
python3 f.py data/abgar/btc/2.csv TIMEFRAME=15 TICKER=\'btcusd\' >> f.txt
python3 f.py data/abgar/btc/3.csv TIMEFRAME=15 TICKER=\'btcusd\' >> f.txt
python3 f.py data/abgar/btc/4.csv TIMEFRAME=15 TICKER=\'btcusd\' >> f.txt

python3 f.py data/abgar/btc/1.csv TIMEFRAME=60 TICKER=\'btcusd\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/btc/2.csv TIMEFRAME=15 TICKER=\'btcusd\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/btc/3.csv TIMEFRAME=15 TICKER=\'btcusd\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/abgar/btc/4.csv TIMEFRAME=15 TICKER=\'btcusd\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
"""
"""
python3 f.py data/ivan/tr_decis_n_0_e_288.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' >> f.txt
python3 f.py data/ivan/tr_decis_n_0_e_330.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' >> f.txt
python3 f.py data/ivan/tr_decis_n_0_e_369.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' >> f.txt
python3 f.py data/ivan/tr_decis_n_1_e_362.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' >> f.txt
python3 f.py data/ivan/tr_decis_n_1_e_491.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' >> f.txt
python3 f.py data/ivan/tr_decis_n_1_e_497.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' >> f.txt
python3 f.py data/ivan/tr_decis_n_2_e_266.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' >> f.txt

python3 f.py data/ivan/tr_decis_n_0_e_288.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_0_e_330.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_0_e_369.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_1_e_362.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_1_e_491.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_1_e_497.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_2_e_266.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
"""

"""
python3 f.py data/ivan/tr_decis_n_2_e_346.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' >> f.txt
python3 f.py data/ivan/tr_decis_n_2_e_393.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' >> f.txt
python3 f.py data/ivan/tr_decis_n_3_e_390.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' >> f.txt
python3 f.py data/ivan/tr_decis_n_3_e_446.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' >> f.txt
python3 f.py data/ivan/tr_decis_n_3_e_487.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' >> f.txt

python3 f.py data/ivan/tr_decis_n_2_e_346.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_2_e_393.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_3_e_390.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_3_e_446.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' CALCULATE_INPUT=fmodels.calculate_input_i CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
python3 f.py data/ivan/tr_decis_n_3_e_487.csv SRC_FORMAT=\'nn\' TIMEFRAME=15 TICKER=\'btcusd\' CALCULATE_INPUT=fmodels.calculate_input_i >> f.txt
"""