import sys
if len(sys.argv) < 2:
	sys.stderr.write( "Use: estimations.py <a-file-with-trades.csv>" )
	sys.exit()

from datetime import datetime, timedelta
from os.path import basename
import numpy as np
import matplotlib.pyplot as plt 
from random import randint
import re 
import eutils, futils

TIMEFRAME = 15 # The time frame.
TICKER = "btcusd" # The ticker.
SRC_FORMAT = 'nn'
DEPOSIT = -1.0
COMPARE_INTEREST_RATE_PCT = 5.0
COMMISSION_PCT=0.0
COMMISSION_ABS=0.0

start_test_date = None
end_test_date = None

candles = None

trades = {}

if re.search( r'\.lst$', sys.argv[1] ): # If a source file passed through command line ends with ".lst"...
	src = futils.read_list_of_files_with_trades( sys.argv[1] )
	if len(src) == 0:
		sys.stderr.write('Error reading the ".lst" file or zero lines read. Exiting...\n')
		sys.exit(0)
else: # A source passed though command line is a file name 
	sys.stderr.write('The first argument must be a ".lst" file. Exiting...\n')
	sys.exit(0)

for a in range(2, len(sys.argv)): # Reading additional parameters
	m = re.match(r'([a-zA-Z0-9\_ ]+=[0-9a-zA-Z \.\'\"\_]+)', sys.argv[a])
	if m:
		exec(m.group(1))

train_test_pairs = [] # Train-test pairs
for s in range(len(src)):
	if 'passed' in src[s]: # Already coupled with something...
		continue
	if src[s]['train_or_test'] == 2: # It's test not train...
		continue 
	file_id = src[s]['file_id']
	for s2 in range(len(src)):
		if s2 == s:
			continue
		if 'passed' in src[s2]:
			continue
		if src[s2]['file_id'] == file_id:
			src[s]['passed'] = True
			src[s2]['passed'] = True
			train_test_pairs.append( [ file_id, src[s]['file_name'], src[s2]['file_name'] ] ) # id, train, test
			break


print_header = True
for p in range(len(train_test_pairs)):
	res = futils.load_trades_and_candles( train_test_pairs[p][1], SRC_FORMAT, TICKER, TIMEFRAME, extra_lookback_candles=1, 
		commission_pct=COMMISSION_PCT, commission_abs=COMMISSION_ABS )
	if res is None:
		sys.stderr.write('Failed to load trades or candles from %s. Skipping...\n' % (train_test_pairs[p][1]))
		continue
	candles = res['candles']
	trades = res['trades']
	e_train = eutils.estimate(candles, trades, TIMEFRAME, COMPARE_INTEREST_RATE_PCT, DEPOSIT)

	res = futils.load_trades_and_candles( train_test_pairs[p][2], SRC_FORMAT, TICKER, TIMEFRAME, extra_lookback_candles=1, 
		commission_pct=COMMISSION_PCT, commission_abs=COMMISSION_ABS )
	if res is None:
		sys.stderr.write('Failed to load trades or candles from %s. Skipping...\n' %(train_test_pairs[p][2]))
		continue
	candles = res['candles']
	trades = res['trades']
	e_test = eutils.estimate(candles, trades, TIMEFRAME, COMPARE_INTEREST_RATE_PCT, DEPOSIT)

	if print_header:
		sys.stdout.write('Ticker, Timeframe, File Name (train), Profit/Loss, Trades Num., Good, Bad, Profit Factor, Profit Factor(%)'\
			', Sharpe, Reward/Risk, Lost Profit')
		sys.stdout.write(', File Name (test), Profit/Loss, Trades Num., Good, Bad, Profit Factor, Profit Factor(%)'\
			', Sharpe, Reward/Risk, Lost Profit\n')
		print_header = False
	sys.stdout.write('%s, %d, %s, %f, %d, %d, %d, %f, %f, %f, %f, %f' % \
		(TICKER, TIMEFRAME, basename(train_test_pairs[p][1]), e_train['pnl']['pnl'][0], e_train['num_trades'], \
		e_train['num_good'], e_train['num_bad'], e_train['pf'], e_train['pf_pct'], e_train['sharpe'], e_train['rr'], e_train['lp']))
	sys.stdout.write(', %s, %f, %d, %d, %d, %f, %f, %f, %f, %f\n' % \
		(basename(train_test_pairs[p][2]), e_test['pnl']['pnl'][0], e_test['num_trades'], \
		e_test['num_good'], e_test['num_bad'], e_test['pf'], e_test['pf_pct'], e_test['sharpe'], e_test['rr'], e_test['lp']))

