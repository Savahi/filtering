import datetime
import re
from pandas import Series  
import numpy as np

def calculate_date_range_of_trades(trades):
	# Searching for min & max... 
	dt_min = trades[0]['time_dt']
	dt_max = trades[0]['closing_time_dt']
	for i in range( 1, len(trades) ):
		if trades[i]['time_dt'] < dt_min:
			dt_min = trades[i]['time_dt']
		if trades[i]['closing_time_dt'] > dt_max:
			dt_max = trades[i]['closing_time_dt']
	return(dt_min, dt_max)


def calculate_pnl( candles, trades, pnl, prefix='_' ):
	len_trades = len(trades)
	len_candles = len(candles['close'])

	return_key = prefix+'return'
	pnl_key = prefix+'pnl'
	time_key = 'time_dt'
	#candles[prefix+'return'] = Series(np.zeros(len_candles), index=candles.index)	
	#candles[prefix+'pnl'] = Series(np.zeros(len_candles), index=candles.index)
	#candles.loc[:,return_key] = 0 # np.zeros(len_candles) # Series(np.zeros(len_candles), index=candles.index)	
	#candles.loc[:,pnl_key] = 0 # np.zeros(len_candles) # Series(np.zeros(len_candles), index=candles.index)
	pnl[return_key] = np.zeros(len_candles)
	pnl[pnl_key] = np.zeros(len_candles)

	if not time_key in pnl:
		pnl[time_key] = [None]*len_candles
		for c in range(len_candles):
			(_, tm) = str_to_time( str(candles['time'][c]) )
			pnl[time_key][c] = tm

	for t in range(len_trades):
		if not 'closing_candle' in trades[t]:
			print("No closing price: position_id=" + str(trades[t]['position_id']))
			input("")
			continue
		c = trades[t]['closing_candle']
		pnl[return_key][c] += trades[t]['profit']

	pnl[pnl_key][len_candles-1] = pnl[return_key][len_candles-1]
	for c in range(len_candles-2,-1,-1):
		pnl[pnl_key][c] += pnl[pnl_key][c+1] + pnl[return_key][c] 


def calculate_potential_returns( candles, trades, pnl, prefix='_' ):
	len_trades = len(trades)
	len_candles = len(candles['close'])

	potential_best_return_key = prefix+'potential_best_return'
	potential_worst_return_key = prefix+'potential_worst_return'
	time_key = 'time_dt'
	pnl[potential_best_return_key] = np.zeros(len_candles)
	pnl[potential_worst_return_key] = np.zeros(len_candles)

	if not time_key in pnl:
		pnl[time_key] = [None]*len_candles
		for c in range(len_candles):
			(_, tm) = str_to_time( str(candles['time'][c]) )
			pnl[time_key][c] = tm

	for t in range(len_trades):
		if not 'closing_candle' in trades[t]:
			print("No closing price: position_id=" + str(trades[t]['position_id']))
			input("")
			continue
		opening_candle = trades[t]['opening_candle']
		closing_candle = trades[t]['closing_candle']
		print('%d - %d' %(opening_candle,closing_candle))
		best_best = 0
		worst_worst = 0
		for c in range(opening_candle,closing_candle-1,-1):
			if trades[t]['side'] == 1:
				best = candles['high'][c] - trades[t]['price']
				worst = candles['low'][c] - trades[t]['price']
				pnl[potential_best_return_key][c] += best
				pnl[potential_worst_return_key][c] += worst
			elif trades[t]['side'] == -1:
				#print( '%f - %f' % (trades[t]['price'], candles['high'][c]) )
				best = trades[t]['price'] - candles['high'][c]
				worst = trades[t]['price'] - candles['low'][c]
				pnl[potential_best_return_key][c] += best
				pnl[potential_worst_return_key][c] += worst
			if best > best_best:
				best_best = best
			if worst_worst < worst:
				worst_worst = worst

		trades[t]['best'] = best_best 
		trades[t]['worst'] = worst_worst 


def calculate_inputs(candles, trades, data, calculate_input, lookback_candles):
	len_trades = len(trades)
	data['inputs'] = [None]*len_trades
	data['outputs'] = [None]*len_trades
	data['time_dt'] = [None]*len_trades
	data['profit'] = [None]*len_trades
	data['profit_pct'] = [None]*len_trades
	data['trade_num'] = [None]*len_trades

	# Making up inputs and outputs 
	for t in range(len(trades)):
		c = trades[t]['opening_candle']
		inp = calculate_input(candles[c+1:c+lookback_candles+1].reset_index())
		data['inputs'][t] = inp
		data['outputs'][t] = []
		data['time_dt'][t] = trades[t]['time_dt']
		data['trade_num'][t] = t
		data['profit'][t] = trades[t]['profit']
		data['profit_pct'][t] = trades[t]['profit_pct']
# end of 


def merge_candles_and_trades(candles, trades):
	len_trades = len(trades)
	len_candles = len(candles['close'])

	for t in range(len_trades):
		for c in range(len_candles-1,-1,-1):
			candle_found = False
			if candles['time'][c] == trades[t]['time']: # The candle related to a trade is found...
				trades[t]['opening_candle'] = c
				trades[t]['opening_at_candle_start'] = True
				candle_found = True
			elif candles['time'][c] > trades[t]['time']:
				trades[t]['opening_candle'] = c+1
				trades[t]['opening_at_candle_start'] = False
				candle_found = True
			if not candle_found:
				continue
				
			for c2 in range(c,-1,-1):
				if candles['time'][c2] == trades[t]['closing_time']: # The candle related to a trade is found...
					trades[t]['closing_candle'] = c2
					trades[t]['closing_at_candle_start'] = True
					break
				if candles['time'][c2] > trades[t]['closing_time']: # The candle related to a trade is found...
					trades[t]['closing_candle'] = c2+1
					trades[t]['closing_at_candle_start'] = False
					break
			break

	for t in range(len_trades):
		if not 'closing_candle' in trades[t]:
			print("NO CLOSING CANDLE HAS BEEN FOUND FOR TRADE ID " + str(trades[t]['position_id']))
			input("PRESS 'ENTER' TO CONTINUE...")
# end of merge_candles_and_trades


def trim_trades(trades, commission_pct=0.0):
	len_trades = len(trades)

	for i in range(len_trades):
		if 'closing_trade' in trades[i]: # If the trade has already been marked as a closing one...
			if trades[i]['closing_trade']: 
				continue # ... 

		position_id = trades[i]['position_id']
		for j in range(i+1,len_trades):
			# if i == j: 
			# 	continue
			if trades[j]['position_id'] == position_id: # The closing trade is found...
				profit = trades[j]['price'] - trades[i]['price']
				if trades[i]['side'] == -1:
					profit = -profit
				if commission_pct > 0.0:
					profit -= np.abs(profit)*(commission_pct/100.0)
				trades[i]['profit'] = profit
				trades[i]['profit_pct'] = (profit * 100.0) / trades[i]['price']				
				trades[i]['closing_price'] = trades[j]['price']
				trades[i]['closing_time'] = trades[j]['time']
				trades[i]['closing_time_seconds'] = trades[j]['time_seconds']
				trades[i]['closing_time_dt'] = trades[j]['time_dt']
				trades[j]['closing_trade'] = True
				break

	i = 0
	while(True):
		if i >= len(trades):
			break
		if 'closing_trade' in trades[i]:
			if trades[i]['closing_trade']:
				del trades[i]
				continue
		if not 'closing_price' in trades[i]: # The trade has not been closed...
			del trades[i]
			continue
		i += 1

	return trades
# the end of trim_trades()	


def read_trades( file_name ):
	trades = []

	file_opened = False
	
	num_lines_read = 0
	num_lines_skipped  = 0

	try:
		file_handle = open(file_name, "r")
		file_opened = True

		for line in file_handle:
			re_line = re.match( r'([ 0-9]+),([ 0-9\.]+),([ 0-9\.]+),([ 0-9]+),([ a-zA-Z]+)', line, re.M|re.I)
		
			if not re_line:
				num_lines_skipped += 1
				continue

			time_str = re_line.group(1)
			(time_seconds, time_dt) = str_to_time( time_str )
			if time_seconds == None:
				num_lines_skipped += 1
				continue

			time = int( int(time_str) / 1000 )

			side = 0
			if re_line.group(5) == "BUY":
				# print("BUY")
				side = 1
			elif re_line.group(5) == "SELL":
				# print("SELL")
				side = -1
			if side == 0:
				num_lines_skipped += 1
				continue

			trades.append( { 'time':time, 'time_str':time_str, 'time_dt':time_dt, 'time_seconds':time_seconds, \
								'price':float(re_line.group(2)), 'volume':float(re_line.group(3)), \
								'position_id':int(re_line.group(4)), 'side':side } )
			num_lines_read += 1
	
	except IOError:
		print( "Error: can\'t find file " + file_name + " or read data" )
	else:
		print( "Lines read = %d, lines skipped = %d" % (num_lines_read, num_lines_skipped) )

	if file_opened:
		file_handle.close()

	return trades
# end of def	


def read_trades_2( file_name ):
	trades = []

	file_opened = False
	
	num_lines_read = 0
	num_lines_skipped  = 0

	position_id = 0
	first_position = True

	try:
		file_handle = open(file_name, "r")
		file_opened = True

		for line in file_handle:
			re_line = re.match( r'([ 0-9]+),([ 0-9]+),([ 0-9]+),([ 0-9\.]+),([ 0-9\.]+),([ 0-9\.]+),([ 0-9]+),([ 0-9]+)', line, re.M|re.I)
			if not re_line:
				num_lines_skipped += 1
				continue

			if int(re_line.group(8)) == 0: # A position hasn't been opened
				continue

			time_str = re_line.group(7)
			(time_seconds, time_dt) = str_to_time( time_str )
			if time_seconds == None:
				num_lines_skipped += 1
				continue

			time = int(time_str)

			side = 0
			if int(re_line.group(3)) == 1:
				# print("BUY")
				side = 1
			elif int(re_line.group(2)) == 1:
				# print("SELL")
				side = -1
			if side == 0:
				num_lines_skipped += 1
				continue

			if not first_position:
				trades.append(  { 'time':time, 'time_str':time_str, 'time_dt':time_dt, 'time_seconds':time_seconds, \
									'price':float(re_line.group(4)), 'volume':1, \
									'position_id':position_id, 'side':side } )	
				position_id += 1
			else:
				first_position = False			
			trades.append(  { 'time':time, 'time_str':time_str, 'time_dt':time_dt, 'time_seconds':time_seconds, \
									'price':float(re_line.group(4)), 'volume':1, \
									'position_id':position_id, 'side':side } )
			#print(position)
			#input("")

			num_lines_read += 1
	
	except IOError:
		print( "Error: can\'t find file " + file_name + " or read data" )
	else:
		print( "Lines read = %d, lines skipped = %d" % (num_lines_read, num_lines_skipped) )

	if file_opened:
		file_handle.close()

	#print(trades)
	#input("")
	return trades
# end of def	

def str_to_time( sDateTime ):
	
	iDateTime = None
	dtDateTime = None

	reDateTime = re.match( r"([0-9][0-9][0-9][0-9])([0-9][0-9])([0-9][0-9])([0-9][0-9])([0-9][0-9])([0-9][0-9])([0-9][0-9][0-9])", sDateTime, re.M|re.I )
	if reDateTime:
		sYear = reDateTime.group(1)
		sMonth = reDateTime.group(2)
		sDay = reDateTime.group(3)
		sH = reDateTime.group(4)
		sM = reDateTime.group(5)
		sS = reDateTime.group(6)
		iYear = int ( sYear )
		iMonth = int ( sMonth )
		iDay = int ( sDay )
		iH = int (sH )
		iM = int (sM)
		iS = int (sS)
		dtDateTime = datetime.datetime( iYear, iMonth, iDay, iH, iM, iS )
		iDateTime = time_to_seconds( dtDateTime )
	else:
		reDateTime = re.match( r"([0-9][0-9][0-9][0-9])([0-9][0-9])([0-9][0-9])([0-9][0-9])([0-9][0-9])([0-9][0-9])", sDateTime, re.M|re.I )
		if reDateTime:
			sYear = reDateTime.group(1)
			sMonth = reDateTime.group(2)
			sDay = reDateTime.group(3)
			sH = reDateTime.group(4)
			sM = reDateTime.group(5)
			sS = reDateTime.group(6)
			iYear = int ( sYear )
			iMonth = int ( sMonth )
			iDay = int ( sDay )
			iH = int (sH )
			iM = int (sM)
			iS = int (sS)
			dtDateTime = datetime.datetime( iYear, iMonth, iDay, iH, iM, iS )
			iDateTime = time_to_seconds( dtDateTime )

	return (iDateTime, dtDateTime)
# end of def

def time_to_seconds( dtDateTime ):
	dtOrigin = datetime.datetime( 1970,1,1,0,0,0 )
	iDateTime = (dtDateTime - dtOrigin).total_seconds()
	iDateTime = int( iDateTime )
	return iDateTime
# end of def


def calc_trades_in_every_bin(data, num_classes):
    train = [0]*num_classes
    test = [0]*num_classes
    train_and_test = [0]*num_classes
    
    for d in range(len(data['train_outputs'])):
        for c in range( num_classes ):
            if data['train_outputs'][d][c] == 1:
                train[c] += 1
    for d in range(len(data['test_outputs'])):
        for c in range( num_classes ):
            if data['test_outputs'][d][c] == 1:
                test[c] += 1
    for c in range(num_classes):
        train_and_test[c] = train[c] + test[c]

    return( train_and_test, train, test)

