#!/bin/bash

python3 e.py data/abgar/btc/1.csv TIMEFRAME=60 >> e.txt
python3 e.py data/abgar/btc/2.csv TIMEFRAME=15 >> e.txt
python3 e.py data/abgar/btc/3.csv TIMEFRAME=15 >> e.txt
python3 e.py data/abgar/btc/4.csv TIMEFRAME=15 >> e.txt

python3 e.py data/abgar/eth/1.csv TIMEFRAME=60 >> e.txt
python3 e.py data/abgar/eth/2.csv TIMEFRAME=15 >> e.txt
python3 e.py data/abgar/eth/3.csv TIMEFRAME=15 >> e.txt
python3 e.py data/abgar/eth/4.csv TIMEFRAME=15 >> e.txt

python3 e.py data/abgar/rts/1.csv TIMEFRAME=60 >> e.txt
python3 e.py data/abgar/rts/2.csv TIMEFRAME=15 >> e.txt
python3 e.py data/abgar/rts/3.csv TIMEFRAME=15 >> e.txt
python3 e.py data/abgar/rts/4.csv TIMEFRAME=15 >> e.txt

python3 e.py data/abgar/si/1.csv TIMEFRAME=60 >> e.txt
python3 e.py data/abgar/si/2.csv TIMEFRAME=15 >> e.txt
python3 e.py data/abgar/si/3.csv TIMEFRAME=15 >> e.txt
python3 e.py data/abgar/si/4.csv TIMEFRAME=15 >> e.txt
