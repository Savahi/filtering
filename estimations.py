import sys
if len(sys.argv) < 2:
	print( "Use: estimations.py <a-file-with-trades.csv>" )
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
from operator import itemgetter
import re 

TIMEFRAME = 30 # The time frame.
TICKER = "btcusd" # The ticker.
SRC_FORMAT = 'platform'
DEPOSIT = 1000000
SHARPE_ITERATION = 48
COMPARE_INTEREST_RATE_PCT = 1.0
COMMISSION_PCT = 0.3

start_test_date = None
end_test_date = None

candles = None

trades = {}

for a in range(2, len(sys.argv)): # Reading additional parameters
	m = re.match(r'([a-zA-Z0-9\_ ]+=[0-9a-zA-Z \.\'\_]+)', sys.argv[a])
	if m:
		exec(m.group(1))

def load_data():
	global candles, trades

	if SRC_FORMAT == 'platform': 
		trades = utils.read_trades(sys.argv[1])
	else:
		trades = utils.read_trades_2(sys.argv[1])
	if len(trades) == 0:
		return False

	trades = utils.trim_trades( trades )
	if len(trades) == 0:
		return False

	(dt_min, dt_max) = utils.calculate_date_range_of_trades(trades)
	dt_start = dt_min - timedelta(minutes = TIMEFRAME)
	dt_end = dt_max + timedelta(minutes=TIMEFRAME*2)
	print( 'DATES INVOLVED: start: %s, end: %s' % ( str(dt_start), str(dt_end) ) )

	# Importing candles...
	candles = import_candles(TICKER, TIMEFRAME, dt_start, dt_end)

	utils.merge_candles_and_trades( candles, trades )

	return True
# end of load_data

load_data() 

num_trades = 0

# Calculating profit factor 
num_bad = 0
num_good = 0
sum_pos = 0
sum_neg = 0
sum_pos_pct = 0
sum_neg_pct = 0
for t in range(len(trades)):
	profit = trades[t]['profit']
	profit_pct = trades[t]['profit_pct']
	if profit > 0:
		sum_pos += profit 
		sum_pos_pct += profit_pct
		num_good += 1
	elif profit < 0:
		sum_neg += -profit
		sum_neg_pct += -profit_pct
		num_bad += 1

if sum_neg > 0:
	pf = sum_pos / sum_neg
	pf_pct = sum_pos_pct / sum_neg_pct
else:
	pf = 100
	pf_pct = 100
print( 'pf = %f(%f), num_good=%d, num_bad=%d' % (pf,pf_pct,num_good,num_bad) )


# Calculating PnL
pnl = {}
utils.calculate_pnl(candles, trades, pnl, "trades_")


# Calculating REWARD-RISK
utils.calculate_potential_returns( candles, trades, pnl, prefix='trades_' )
risk = 0
for t in range(len(trades)):
	if 'worst' in trades[t]:
		risk += np.abs( trades[t]['profit'] - trades[t]['worst'] )
reward = sum_pos - sum_neg
if risk > 0:
	rr = reward / risk # ... calculating reward-risk
	print('Reward/Risk Ratio = %f' % (rr))
else:
	if reward > 0:
		print("No risk has been experienced throughout the trading session!")
	else:
		print("Can't calculate reward-risk!")


# Calculating SHARPE 
pct_a_day = (1.0 + COMPARE_INTEREST_RATE_PCT / 100.0 / 365.0) # Daily risk-free interest 
candles_in_a_day = int(60 * 24 / TIMEFRAME) # The number of candles making up a day
actual_pnl = [ DEPOSIT ] # PnL at every candle
compare_pnl_with = [ DEPOSIT ] # Risk-free income at every candle
pnl_diffs = [0] # 
actual_return = [ 0 ]
compare_return_with = DEPOSIT * pct_a_day
return_diffs = [0]
time = [ pnl['time_dt'][-1] ]
for c in range( len(pnl['trades_pnl'])-candles_in_a_day-1, -1, -candles_in_a_day ):
	new_compare_pnl_with = compare_pnl_with[-1] * pct_a_day
	compare_pnl_with.append( new_compare_pnl_with )
	new_actual_pnl = DEPOSIT + pnl['trades_pnl'][c]
	actual_pnl.append(new_actual_pnl)
	pnl_diffs.append(new_actual_pnl - new_compare_pnl_with)

	actual_return.append(pnl['trades_return'][c])
	return_diffs.append(pnl['trades_return'][c] - compare_return_with)

	time.append(pnl['time_dt'][c] )

print(pnl_diffs)

std = np.std( pnl_diffs )
if std > 0:
	sharpe = np.mean( pnl_diffs ) / std
else:
	sharpe = float('NaN')
print('sharpe=%f'%(sharpe))

# How many stds in average trade 
profit_by_trade = []
for t in range(len(trades)):
	profit_by_trade.append(trades[t]['profit'])

std = np.std(profit_by_trade)
if std > 0:
	num_stds_in_average_trade = np.mean(profit_by_trade) / std
else:
	num_stds_in_average_trade = float('NaN')
print( 'Num. stds in average trade = %f' % (num_stds_in_average_trade) )

import matplotlib.pyplot as plt
plt.plot(time, actual_pnl)
plt.plot(time, compare_pnl_with)
plt.show()
