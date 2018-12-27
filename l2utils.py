import sys
import re
import numpy as np
from datetime import datetime, timedelta
from tradingene.data.load import import_candles
from futils import str_to_time

def load_and_prepare_data(src, ticker, timeframe, lookahead_candles=0, class_threshold=1.0, num_classes=3, train_or_test=1):

	all_signals = []
	time_limits = { 'min_time':None, 'max_time':None }
	for s in range(len(src)):
		if int(src[s]['train_or_test']) != train_or_test:
			continue

		file_name = src[s]['file_name']		
		signals = read_signals( file_name, time_limits=time_limits )
		if len(signals) == 0:
			return None
		all_signals.append( { 'id':s, 'signals':signals } )
	len_all_signals = len(all_signals)
	if len_all_signals == 0:
		return None

	# Calculating starting and ending dates
	dt_start = time_limits['min_time'] - timedelta( minutes = timeframe * 1 )
	dt_end = time_limits['max_time'] + timedelta( minutes = timeframe * (lookahead_candles + 1) )

	sys.stderr.write( 'DATES INVOLVED: start: %s, end: %s\n' % ( str(dt_start), str(dt_end) ) )

	# Importing candles...
	candles = import_candles(ticker, timeframe, dt_start, dt_end)
	len_candles = len(candles)
	
	seconds = [0]*len_candles
	datetimes = [0]*len_candles
	for c in range(len_candles):
		seconds[c], datetimes[c] = str_to_time( str(candles['time'][c]), round_to_minutes=True )

	# Mergin all singles
	inputs = []
	outputs = []
	candle_refs = []
	num_samples_by_classes = [0]*num_classes
	for c in range(lookahead_candles, len_candles): # For each candle...
		inputs_row = []
		inputs_appended = 0
		for a in range(len_all_signals):    		# For each set of signals in "all_signals"...
			signals = all_signals[a]['signals'] 	# ...getting a set of signals
			for s in range(len(signals)):			# For each output in the set of signals...
				if seconds[c] == signals[s]['time_seconds']: #... if the one found that corresponds to the candle currently iterated...
					for ioutput in range(signals[s]['num_outputs']):  # For all the outputs the net gives for this time...
						inputs_row.append( signals[s]['output'+str(ioutput)] ) 	# ...appending each to make up eventually a great new input
					inputs_appended += 1										# for another upper level (l2) network 
					break

		if inputs_appended == len_all_signals:
			sum_of_price_values = 0.0
			for c2 in range(c, c - lookahead_candles - 1, -1):
				sum_of_price_values += candles['high'][c2]
				sum_of_price_values += candles['low'][c2]
				sum_of_price_values += candles['close'][c2]
			average_price_value = sum_of_price_values / (3.0 * (lookahead_candles+1))

			if average_price_value > candles['open'][c] * (1.0 + class_threshold/100.0):
				if num_classes == 3:
					output = [0,0,1]
					num_samples_by_classes[2] += 1
				else:
					output = [0,1]
					num_samples_by_classes[1] += 1
			elif average_price_value < candles['open'][c] * (1.0 - class_threshold/100.0):
				if num_classes == 3:
					output = [1,0,0]
					num_samples_by_classes[0] += 1
				else:
					output = [1,0]
					num_samples_by_classes[0] += 1
			else:
				if num_classes == 3:
					output = [0,1,0]
					num_samples_by_classes[1] += 1
				else:
					output = None

			if output is not None:
				inputs.append(inputs_row)
				candle_refs.append(c)
				outputs.append(output)

	if len(inputs) == 0:
		return None

	inputs = np.array(inputs)
	outputs = np.array(outputs)

	# end of "for..."
	return { 'candles':candles, 'seconds':seconds, 'datetimes':datetimes, 'inputs':inputs, 'outputs':outputs, 
		'candle_refs':candle_refs, 'num_samples_by_classes':num_samples_by_classes }
# end of "load_and_prepare_data"


def read_signals( file_name, time_limits=None ):
	signals = []

	file_opened = False
	
	num_lines_read = 0
	num_lines_skipped  = 0

	try:
		file_handle = open(file_name, "r")
		file_opened = True

		for line in file_handle:
			re_line = re.match( r'([ 0-9]+),([ 0-9]+),([ 0-9]+),([ 0-9\.]+),([ 0-9\.]+),([ 0-9\.]+),([ 0-9]+),([ 0-9\.\;\-\e]+)', line, re.M|re.I)
			if not re_line:
				num_lines_skipped += 1
				continue

			time_str = re_line.group(7)
			(time_seconds, time_dt) = str_to_time( time_str, round_to_minutes=True )
			if time_seconds == None:
				num_lines_skipped += 1
				continue

			time = int(time_str)
			if time_limits is not None:
				if time_limits['min_time'] is None:
					time_limits['min_time'] = time_dt 
				elif time_dt < time_limits['min_time']:
					time_limits['min_time'] = time_dt
				if time_limits['max_time'] is None:
					time_limits['max_time'] = time_dt
				elif time_dt > time_limits['max_time']:
					time_limits['max_time'] = time_dt

			net_output = re_line.group(8) 	# Getting net output
			isdecimalpoint = net_output.find('.') 	# Searching for a decimal point to know what kind of output is it...
			issemicolon = net_output.find(';')	# Searching for a decimal point to know what kind of output is it...
			if isdecimalpoint == -1 and issemicolon == -1:  	# 0/1 kind of output
				if int(net_output) == 0: # A position hasn't been opened, the prior signal is in effect now
					signal = 0
				else:
					if int(re_line.group(3)) == 1: # Buy?
						signal = 1 
					elif int(re_line.group(2)) == 1: # Sell?
						signal = -1
					else:
						signal = 0
				signals.append( { 'time_seconds':time_seconds, 'time_dt':time_dt, 'num_outputs':1, 'output0':signal } )
			else:		# Net probabilities serve as output
				outputs = re.split(';', net_output)
				new_signal = { 'time_seconds':time_seconds, 'time_dt':time_dt, 
					'actual_output':int(re_line.group(6)), 'num_outputs':len(outputs) }
				for ioutput in range(len(outputs)):
					new_signal['output'+str(ioutput)] = float(outputs[ioutput])
				signals.append( new_signal )
			num_lines_read += 1
	except IOError:
		sys.stderr.write( "Error: can\'t find file " + file_name + " or read data.\n" )
	else:
		sys.stderr.write( "Lines read = %d, lines skipped = %d.\n" % (num_lines_read, num_lines_skipped) )

	if file_opened:
		file_handle.close()

	return signals
# end of def	

