# eutils.py
import sys
import numpy as np
from futils import calculate_pnl, calculate_potential_returns

def estimate(candles, trades, timeframe, compare_interest_rate_pct=1.0, initial_deposit=-1, prefix=''):

	basic_stat = calculate_basic_stat( trades )

	# Calculating PnL...
	pnl = {}
	calculate_pnl( candles, trades, pnl, prefix )

	# Calculating potential profits and losses...
	calculate_potential_returns( candles, trades, pnl, prefix )

	# Calculating reward/risk ratio...
	reward = basic_stat['sum_pos'] - basic_stat['sum_neg']
	rr = calculate_reward_risk( candles, trades, pnl, reward )

	# Calculating lost profit overall...
	lp = calculate_lost_profit( trades )

	# Calculating initial deposit if required (to calculate Sharpe next)...
	if initial_deposit is None or initial_deposit < 0.0:
		initial_deposit = calculate_initial_deposit( trades, pnl, prefix )

	# Calculating sharpe ratio...
	sharpe = calculate_sharpe( pnl, compare_interest_rate_pct, timeframe, initial_deposit, prefix )

	basic_stat['rr'] = rr
	basic_stat['lp'] = lp
	basic_stat['sharpe'] = sharpe
	basic_stat['pnl'] = pnl

	return basic_stat
# end of def


def calculate_basic_stat( trades ):
	basic_stat={} 
	
	len_trades = len(trades)
	num_bad = 0
	num_good = 0
	sum_pos = 0
	sum_neg = 0
	sum_pos_pct = 0
	sum_neg_pct = 0
	for t in range(len_trades):
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
	elif sum_pos > 0:
		pf = 100.0
		pf_pct = 100.0
	else:
		pf = 0.0
		pf_pct = 0.0
	sys.stderr.write( 'pf = %f(%f), num_good=%d, num_bad=%d\n' % (pf,pf_pct,num_good,num_bad) )

	basic_stat['pf'] = pf
	basic_stat['pf_pct'] = pf_pct
	basic_stat['num_good'] = num_good
	basic_stat['num_bad'] = num_bad
	basic_stat['num_trades'] = num_good + num_bad
	basic_stat['sum_pos'] = sum_pos
	basic_stat['sum_neg'] = sum_neg
	basic_stat['sum_pos_pct'] = sum_pos_pct
	basic_stat['sum_neg_pct'] = sum_neg_pct
	return basic_stat 
# end of def


def calculate_initial_deposit( trades, pnl, prefix='' ):
	len_trades = len(trades)	
	initial_deposit = trades[0]['price']
	for t in range(1,len_trades):
		c = trades[t]['opening_candle']
		cur_pnl = pnl[prefix + 'pnl'][c]
		initial_deposit_plus_pnl = initial_deposit + cur_pnl
		if initial_deposit_plus_pnl < trades[t]['price']:
			initial_deposit += trades[t]['price'] - initial_deposit_plus_pnl
	return float(initial_deposit)
# end of def 


def calculate_sharpe( pnl, compare_interest_rate_pct, timeframe, initial_deposit, prefix='' ):
	# Calculating SHARPE 
	pct_a_day = (1.0 + compare_interest_rate_pct / 100.0 / 365.0) # Daily risk-free interest 
	candles_in_a_day = int(60 * 24 / timeframe) # The number of candles making up a day
	actual_pnl = [ initial_deposit ] # PnL at every candle
	compare_pnl_with = [ initial_deposit ] # Risk-free income at every candle
	pnl_diffs = [0] # 
	actual_return = [ 0 ]
	compare_return_with = initial_deposit * pct_a_day
	return_diffs = [0]
	time = [ pnl['time_dt'][-1] ]
	for c in range( len(pnl[prefix+'pnl']) - candles_in_a_day - 1, -1, -candles_in_a_day ):
		new_compare_pnl_with = compare_pnl_with[-1] * pct_a_day
		compare_pnl_with.append( new_compare_pnl_with )
		new_actual_pnl = initial_deposit + pnl[prefix + 'pnl'][c]
		actual_pnl.append(new_actual_pnl)
		pnl_diffs.append(new_actual_pnl - new_compare_pnl_with)

		actual_return.append( pnl[prefix+'return'][c] )
		return_diffs.append( pnl[prefix+'return'][c] - compare_return_with )

		time.append(pnl['time_dt'][c] )

	std = np.std( pnl_diffs )
	if std > 0:
		sharpe = np.mean( pnl_diffs ) / std
	else:
		sharpe = float('NaN')
	sys.stderr.write('sharpe=%f\n'%(sharpe))
	return sharpe
# end of def



def calculate_reward_risk( candles, trades, pnl, reward ):
	len_trades = len(trades)	
	risk = 0
	for t in range(len_trades):
		if 'worst' in trades[t]:
			if trades[t]['profit'] > 0:
				risk += trades[t]['worst']
			elif trades[t]['worst'] > -trades[t]['profit']:
				risk += (trades[t]['worst'] + trades[t]['profit'])
	if risk > 0:
		rr = reward / risk # ... calculating reward-risk
		sys.stderr.write('Reward/Risk Ratio = %f\n' % (rr))
	else:
		if reward > 0:
			rr = 1e10
			sys.stderr.write("No risk has been experienced throughout the trading session!\n")
		else:
			rr = float('NaN')
			sys.stderr.write("Can't calculate reward-risk!\n")
	return rr
# end of def


# Calculating LOST Profit
def calculate_lost_profit( trades ):
	len_trades = len(trades)	
	lost_profit = 0
	for t in range(len_trades):
		if 'best' in trades[t]:
			if trades[t]['profit'] < 0:
				lost_profit += trades[t]['best']
			elif trades[t]['best'] > trades[t]['profit']:
				lost_profit += (trades[t]['best'] - trades[t]['profit'])
	sys.stderr.write('Lost Profit = %f\n' % (lost_profit))
	return lost_profit
# end of def


