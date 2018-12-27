import sys
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from tradingene.data.load import import_candles
import re
import keras
from pandas import Series  
import numpy as np
from random import randint

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


def calculate_pnl( candles, trades, pnl, prefix='' ):
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
			(_, tm) = str_to_time( str( candles['time'].iloc[c] ) )
			pnl[time_key][c] = tm

	for t in range(len_trades):
		if not 'closing_candle' in trades[t]:
			sys.stderr.write("No closing price: position_id=" + str(trades[t]['position_id']) + ".\n")
			continue
		c = trades[t]['closing_candle']
		pnl[return_key][c] += trades[t]['profit']

	pnl[pnl_key][len_candles-1] = pnl[return_key][len_candles-1]
	for c in range(len_candles-2,-1,-1):
		pnl[pnl_key][c] += pnl[pnl_key][c+1] + pnl[return_key][c] 


def calculate_potential_returns( candles, trades, pnl, prefix='' ):
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
			sys.stderr.write("No closing price: position_id=" + str(trades[t]['position_id']) + ".\n")
			continue
		opening_candle = trades[t]['opening_candle']
		closing_candle = trades[t]['closing_candle']
		#print('o/c: %s, %s' % (candles['open'][opening_candle], candles['open'][closing_candle]))
		best_best = 0
		worst_worst = 0
		for c in range(opening_candle,closing_candle,-1): # Not counting the latest candle
			if trades[t]['side'] == 1:
				best = candles['high'].iloc[c] - trades[t]['price']
				worst = trades[t]['price'] - candles['low'].iloc[c]
				#print('worst=%f'%(worst))
				pnl[potential_best_return_key][c] += best
				pnl[potential_worst_return_key][c] += worst
			elif trades[t]['side'] == -1:
				best = trades[t]['price'] - candles['low'].iloc[c]
				worst = candles['high'].iloc[c] - trades[t]['price'] 
				#print('worst=%f'%(worst))
				pnl[potential_best_return_key][c] += best
				pnl[potential_worst_return_key][c] += worst
			if best > best_best:
				best_best = best
			if worst > worst_worst:
				worst_worst = worst

		if trades[t]['profit'] > 0 and best_best < trades[t]['profit']: # Not counting latest candle may cause best_best being underestimated
			best_best = trades[t]['profit']
		if trades[t]['profit'] < 0 and worst_worst < -trades[t]['profit']: # Not counting the latest candle may cause worst_worst being underestimated
			worst_worst = -trades[t]['profit']
		trades[t]['best'] = best_best 
		trades[t]['worst'] = worst_worst 


def calculate_all_inputs(candles, trades, data, calculate_input, lookback_candles):
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

	ctimes = np.array(candles['time'], dtype=np.uint64)
	for t in range(len_trades):
		sys.stderr.write('Merging trades: %d/%d\r' % (t,len_trades))
		# candle_found = False
		indexes = np.argwhere( ctimes >= trades[t]['time'] )
		if len(indexes) == 0:
			continue
		c = indexes[-1][0]
		if ctimes[c] == trades[t]['time']: # The candle related to a trade is found...
			trades[t]['opening_candle'] = c
			trades[t]['opening_at_candle_start'] = True
		elif ctimes[c] > trades[t]['time']:
			if c < len_candles-1:
				trades[t]['opening_candle'] = c+1
			else:
				trades[t]['opening_candle'] = c
			trades[t]['opening_at_candle_start'] = False

		indexes2 = np.argwhere( ctimes >= trades[t]['closing_time'] )
		if len(indexes2) == 0:
			continue
		c2 = indexes2[-1][0]
		if ctimes[c2] == trades[t]['closing_time']: # The candle related to a trade is found...
			trades[t]['closing_candle'] = c2
			trades[t]['closing_at_candle_start'] = True
		elif ctimes[c2] > trades[t]['closing_time']: # The candle related to a trade is found...
			if c2 < len_candles-1:
				trades[t]['closing_candle'] = c2+1
			else:
				trades[t]['closing_candle'] = c2						
			trades[t]['closing_at_candle_start'] = False
		
		'''
		if candles['time'].iloc[c] == trades[t]['time']: # The candle related to a trade is found...
			trades[t]['opening_candle'] = c
			trades[t]['opening_at_candle_start'] = True
			candle_found = True
		elif candles['time'].iloc[c] > trades[t]['time']:
			if c < len_candles-1:
				trades[t]['opening_candle'] = c+1
			else:
				trades[t]['opening_candle'] = c
			trades[t]['opening_at_candle_start'] = False
			candle_found = True
		if not candle_found:
			continue
			
		for c2 in range(c,-1,-1):
			if candles['time'].iloc[c2] == trades[t]['closing_time']: # The candle related to a trade is found...
				trades[t]['closing_candle'] = c2
				trades[t]['closing_at_candle_start'] = True
				break
			if candles['time'].iloc[c2] > trades[t]['closing_time']: # The candle related to a trade is found...
				if c2 < len_candles-1:
					trades[t]['closing_candle'] = c2+1
				else:
					trades[t]['closing_candle'] = c2						
				trades[t]['closing_at_candle_start'] = False
				break
		break
		'''
	for t in range(len_trades):
		if not 'closing_candle' in trades[t]:
			sys.stderr.write("NO CLOSING CANDLE HAS BEEN FOUND FOR TRADE ID " + str(trades[t]['position_id']) + ".\n")
# end of merge_candles_and_trades


def trim_trades(trades, train_or_test=0, src_id=0, commission_pct=0.0, commission_abs=0.0):
	len_trades = len(trades)

	for i in range(len_trades):
		if 'closing_trade' in trades[i]: # If the trade has already been marked as a closing one...
			if trades[i]['closing_trade']: 
				continue # ... 
		if 'profit' in trades[i]: # If the trade has already been trimmed...
			continue

		position_id = trades[i]['position_id']
		for j in range(i+1,len_trades): # Searching for the closing trade...
			# if i == j: 
			# 	continue
			if trades[j]['position_id'] == position_id: # The closing trade is found...
				profit = trades[j]['price'] - trades[i]['price']
				if trades[i]['side'] == -1:
					profit = -profit
				if commission_pct > 0.0:
					profit -= np.abs(profit)*(commission_pct/100.0)
				elif commission_abs > 0.0:
					profit -= commission_abs
				trades[i]['profit'] = profit
				trades[i]['profit_pct'] = (profit * 100.0) / trades[i]['price']				
				trades[i]['closing_price'] = trades[j]['price']
				trades[i]['closing_time'] = trades[j]['time']
				trades[i]['closing_time_seconds'] = trades[j]['time_seconds']
				trades[i]['closing_time_dt'] = trades[j]['time_dt']
				trades[i]['train_or_test'] = train_or_test
				trades[i]['src_id'] = src_id
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


def read_trades( file_name, trades=None ):
	if trades is None:
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
				side = 1
			elif re_line.group(5) == "SELL":
				side = -1
			if side == 0:
				num_lines_skipped += 1
				continue

			trades.append( { 'time':time, 'time_str':time_str, 'time_dt':time_dt, 'time_seconds':time_seconds, \
								'price':float(re_line.group(2)), 'volume':float(re_line.group(3)), \
								'position_id':int(re_line.group(4)), 'side':side } )
			num_lines_read += 1
	
	except IOError:
		sys.stderr.write( "Error: can\'t find file " + file_name + " or read data.\n" )
	else:
		sys.stderr.write( "Lines read = %d, lines skipped = %d.\n" % (num_lines_read, num_lines_skipped) )

	if file_opened:
		file_handle.close()

	return trades
# end of def	


def read_trades_2( file_name, trades=None, last_positionId=-1 ):
	if trades is None:
		trades = []

	file_opened = False
	
	num_lines_read = 0
	num_lines_skipped  = 0

	position_id = last_positionId+1
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
				side = 1
			elif int(re_line.group(2)) == 1:
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
			num_lines_read += 1
	
	except IOError:
		sys.stderr.write( "Error: can\'t find file " + file_name + " or read data.\n" )
	else:
		sys.stderr.write( "Lines read = %d, lines skipped = %d.\n" % (num_lines_read, num_lines_skipped) )

	if file_opened:
		file_handle.close()

	return trades, position_id
# end of def	

def str_to_time( sDateTime, round_to_minutes=False ):
	
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
		if not round_to_minutes:
			iS = int (sS)
		else:
			iS = 0
		dtDateTime = datetime( iYear, iMonth, iDay, iH, iM, iS )
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
			if not round_to_minutes:
				iS = int (sS)
			else:
				iS = 0
			dtDateTime = datetime( iYear, iMonth, iDay, iH, iM, iS )
			iDateTime = time_to_seconds( dtDateTime )

	return (iDateTime, dtDateTime)
# end of def

def time_to_seconds( dtDateTime ):
	dtOrigin = datetime( 1970,1,1,0,0,0 )
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


def load_trades_and_candles( src, src_format, ticker, timeframe, 
	extra_lookback_candles=0, extra_lookahead_candles=0, commission_pct=0.0, commission_abs=0.0, splitSides=False ):
	if isinstance( src, str ):
		src = [ {'file_name':src, 'train_or_test':0 } ] # "1" stands for TRAIN, "2" stands for TEST, 0" stands for UNDEFINED

	trades = []
	position_id = -1 # For "read_trades_2" position_id is required if a set of files with trades has been loaded
	for s in range(len(src)):
		file_name = src[s]['file_name']
		train_or_test = int(src[s]['train_or_test'])
		if src_format == 'platform': 
			trades = read_trades(file_name, trades)
		else:
			trades, poition_id = read_trades_2(file_name, trades, position_id)
		if not 'file_id' in src[s]:
			src_id = s
		else:
			src_id = int(src[s]['file_id'])
		trades = trim_trades( trades, src_id=src_id, train_or_test=train_or_test, 
			commission_pct=commission_pct, commission_abs=commission_abs )			
	if len(trades) == 0:
		return None

	# Searching for min & max... 
	dt_min = trades[0]['time_dt']
	dt_max = trades[0]['closing_time_dt']
	for i in range( 1, len(trades) ):
		if trades[i]['time_dt'] < dt_min:
			dt_min = trades[i]['time_dt']
		if trades[i]['closing_time_dt'] > dt_max:
			dt_max = trades[i]['closing_time_dt']

	# Calculating train and test starting and ending dates
	dt_start = dt_min - timedelta(minutes = timeframe * (extra_lookback_candles + 1))
	dt_end = dt_max + timedelta(minutes=timeframe*(2+extra_lookahead_candles))

	sys.stderr.write( 'DATES INVOLVED: start: %s, end: %s\n' % ( str(dt_start), str(dt_end) ) )

	# Importing candles...
	candles = import_candles(ticker, timeframe, dt_start, dt_end)

	merge_candles_and_trades( candles, trades )

	return { 'candles':candles, 'trades':trades, 'dt_min':dt_min, 'dt_max':dt_max, 'dt_start':dt_start, 'dt_end':dt_end }
# end of load_data


def calculate_data_and_train_models(candles, trades, create_model_fn, calculate_input_fn, threshold_abs=0.0, threshold_pct=0.0, \
			train_vs_test=None, num_classes=None, use_weights_for_classes=False, use_2_classes_with_random_pick=False, 
			num_models=1, num_epochs=1000, verbose=False, filter_by_side=None, activation=None, optimizer=None, 
			units_multiplier=None, samples_multiplier=None, num_hidden_layers=None):
	data = {}
	data['train_inputs'] = []
	data['train_outputs'] = []
	data['train_profit'] = []
	data['train_profit_pct'] = []
	data['train_trade_num'] = []
	data['test_inputs'] = []
	data['test_outputs'] = []
	data['test_profit'] = []
	data['test_profit_pct'] = []
	data['test_trade_num'] = []

	lookback_candles, num_features = calculate_input_fn('query_input')
	calculate_all_inputs( candles, trades, data, calculate_input_fn, lookback_candles )

	# Searching for "good" and "bad" trades... 
	num_bad = 0
	num_good = 0
	output_good = [0,1]
	output_bad = [1,0]
	for t in range(len(data['trade_num'])): # "trade_num" is the index in the "trades" array
		profit = data['profit'][t]
		profit_pct = data['profit_pct'][t]
		if profit > threshold_abs and profit_pct > threshold_pct:
			data['outputs'][t] = output_good
			num_good += 1
		else:
			data['outputs'][t] = output_bad
			num_bad += 1

	sys.stderr.write('num_good=%d, num_bad=%d\n' % (num_good,num_bad) )

	if use_weights_for_classes or use_2_classes_with_random_pick: # With weights or random pick only 2 classes are allowed...
		num_classes = 2
	else:
		if num_classes is None:
			num_classes = int( float(num_bad) / float(num_good) + 0.9 ) + 1			
		sys.stderr.write("num_classes=%d\n" % (num_classes))
		if num_classes > 1:
			trades_sorted = [ x for _, x in sorted(zip( data['profit_pct'], data['trade_num'] )) ] # Trade numbers sorted by profit 
			for b in range(num_classes): # For each bin 
				startIndex = int( b * (num_good + num_bad) / (num_classes) ) # Starting and ending indexes within the trades_sorted array
				endIndex = int( (b+1) * (num_good + num_bad) / (num_classes) )

				output = [] 
				for i in range( num_classes ):
					if i == b:
						output.append(1)
					else:
						output.append(0)

				for i in range( startIndex, endIndex ):
					trade_num = trades_sorted[i] #
					for t in range(len(data['trade_num'])): # Searching for train data related to "trade_num" trade...
						if data['trade_num'][t] == trade_num:
							data['outputs'][t] = output
							break
		else:
			sys.stderr.write("ACCORDING TO THE SETTINGS SPECIFIED THERE IS ONLY ONE CLASS IN THE SAMPLE SET. Exiting...\n")
			return None
	# end of "if"

	# Splitting for train and test
	len_data = len(data['trade_num'])
	if train_vs_test is not None:  
		last_train_trade = int(len_data * (1.0 - train_vs_test))
		sys.stderr.write('Data size = %d (train = %d)' % (len_data,last_train_trade))
	else:
		last_train_trade = None
		sys.stderr.write('Data size = %d' % (len_data))

	for t in range(len_data):
		if filter_by_side is not None: 			# If filtering by side is required...
			trade_num = data['trade_num'][t]
			if trades[trade_num]['side'] != filter_by_side:
				continue

		if last_train_trade is not None:
			if data['trade_num'][t] > last_train_trade:
				train_or_test = 2
			else:
				train_or_test = 1
		else:
			trade_num = data['trade_num'][t]
			train_or_test = trades[trade_num]['train_or_test']

		if train_or_test == 1: # Train
			data['train_inputs'].append(data['inputs'][t])
			data['train_outputs'].append(data['outputs'][t])
			data['train_profit'].append(data['profit'][t])
			data['train_profit_pct'].append(data['profit_pct'][t])
			data['train_trade_num'].append(data['trade_num'][t])
		else: # Test
			data['test_inputs'].append(data['inputs'][t])
			data['test_outputs'].append(data['outputs'][t])
			data['test_profit'].append(data['profit'][t])
			data['test_profit_pct'].append(data['profit_pct'][t])
			data['test_trade_num'].append(data['trade_num'][t])
	# end of "for"

	if len(data['train_inputs']) <= 1 or len(data['test_inputs']) < 1:
		return None

	data['train_inputs'] = np.array(data['train_inputs'])
	data['train_outputs'] = np.array(data['train_outputs'])
	data['test_inputs'] = np.array(data['test_inputs'])
	data['test_outputs'] = np.array(data['test_outputs'])

	#scaler = StandardScaler() # Creating an instance of a scaler.
	#scaler.fit(data['train_inputs']) # Fitting the scaler.
	#data['train_inputs'] = scaler.transform(data['train_inputs']) # Normalizing data
	#data['test_inputs'] = scaler.transform(data['test_inputs']) # Normalizing data
	scaler=None

	if units_multiplier is not None:
		multiplier = units_multiplier
	elif samples_multiplier is not None:
		multiplier = int( float(len(data['train_inputs'])) * float(samples_multiplier) )
	else:
		multiplier = None

	models = []
	if use_weights_for_classes:
		inputs = data['train_inputs']
		outputs = data['train_outputs']
		num_trades = num_good + num_bad
		for m in range(num_models):
			cw = { 0: num_good*100.0/num_trades, 1: num_bad*100.0/num_trades }
			model = create_model_fn(num_features, num_classes, 
				activation=activation, optimizer=optimizer, 
				units_multiplier=multiplier, num_hidden_layers=num_hidden_layers) # Creating a model
			model.fit(inputs, outputs, epochs=num_epochs, class_weight=cw, verbose=verbose)
			models.append( model )
	elif use_2_classes_with_random_pick:
		num_models = int( float(num_bad) / float(num_good) + 0.9 )
		for m in range(num_models):
			if num_models > 1:
				inputs = np.array(data['train_inputs'])
				outputs = np.array(data['train_outputs'])
				len_outputs = np.shape(data['train_outputs'])[0]
				for i in range(num_good):
					# random pick
					while(True):
						random_pick = randint(0, len_outputs-1)
						if data['train_outputs'][random_pick][1] == 1:
							continue
						inputs = np.delete(inputs, (random_pick), axis=0)
						outputs = np.delete(outputs, (random_pick), axis=0)
						len_outputs -= 1
						break
			else:
				inputs = data['train_inputs']
				outputs = data['train_outputs']
			model = create_model_fn(num_features, num_classes, 
				activation=activation, optimizer=optimizer, 
				units_multiplier=multiplier, num_hidden_layers=num_hidden_layers) # Creating a model
			model.fit( inputs, outputs, epochs=num_epochs, verbose=verbose ) # Training the model
			models.append( model )
	else:
		inputs = data['train_inputs']
		outputs = data['train_outputs']
		for m in range(num_models):
			model = create_model_fn(num_features, num_classes, 
				optimizer=optimizer, activation=activation,  
				units_multiplier=multiplier, num_hidden_layers=num_hidden_layers) # Creating a model
			model.fit( inputs, outputs, epochs=num_epochs, verbose=verbose ) # Training the model
			models.append( model )

	return { 'data':data, 'scaler':scaler, 'models':models, 'num_classes':num_classes, 'num_good':num_good, 'num_bad':num_bad }
# end of load_data


def read_list_of_files_with_trades( file_name ):
	#sys.stderr.write( "Reading list of files with trades...\n" )

	list_of_files = []

	file_opened = False
	
	num_lines_read = 0
	num_lines_skipped  = 0

	try:
		file_handle = open(file_name, "r")
		file_opened = True

		for line in file_handle:
			re_line = re.match( r'^ *([a-zA-Z0-9\/\_\-\.]+) *\t *([0-9])[ \t\r\n]*$', line, re.M|re.I ) # File name \t train_or_test
			if re_line:
				list_of_files.append( { 'file_name':re_line.group(1), 'train_or_test': int(re_line.group(2)) } )
				num_lines_read += 1
			else:
				re_line = re.match( r'^ *([a-zA-Z0-9\/\_\-\.]+)[ \t\r\n]*$', line, re.M|re.I ) # File name only
				if re_line:
					list_of_files.append( { 'file_name':re_line.group(1), 'train_or_test': 0 } )
					num_lines_read += 1
				else: # File name \t id \t train_or_test ?
					re_line = re.match( r'^ *([a-zA-Z0-9\/\_\-\.]+) *\t *([0-9]+) *\t *([0-9])[ \t\r\n]*$', line, re.M|re.I ) 
					if re_line:
						list_of_files.append( { 'file_name':re_line.group(1), \
							'file_id':int(re_line.group(2)), \
							'train_or_test': int(re_line.group(3)) } )
						num_lines_read += 1
					else:
						num_lines_skipped += 1
	except IOError:
		sys.stderr.write( "Error: can\'t find file " + file_name + " or read data.\n" )
	else:
		sys.stderr.write( "Lines read = %d, lines skipped = %d.\n" % (num_lines_read, num_lines_skipped) )

	if file_opened:
		file_handle.close()

	return list_of_files
# end of def	
