import sys
import utils

def load_candles_and_trades(trades_file, trades_file_format, ticker, timeframe, ):

	if trades_file_format == 'platform': 
		trades = utils.read_trades(trades_file)
	else:
		trades = utils.read_trades_2(trades_file)
	if len(trades) == 0:
		return False

	trades = utils.trim_trades( trades )
	if len(trades) == 0:
		return False

	# Searching for min & max... 
	dt_min = trades[0]['time_dt']
	dt_max = trades[0]['closing_time_dt']
	for i in range( 1, len(trades) ):
		if trades[i]['time_dt'] < dt_min:
			dt_min = trades[i]['time_dt']
		if trades[i]['closing_time_dt'] > dt_max:
			dt_max = trades[i]['closing_time_dt']

	# Calculating train and test starting and ending dates
	dt_start = dt_min - timedelta(minutes = timeframe*(look_back+1))
	dt_end = dt_max + timedelta(minutes=timeframe*2)
	num_days = (dt_max - dt_min).days
	num_days_in_test = int(num_days * train_vs_test)
	dt_train_end = dt_min + timedelta(num_days_in_test)

	sys.stderr.write( 'DATES INVOLVED: start: %s, end: %s, train end: %s' % ( str(dt_start), str(dt_end), str(dt_train_end) ) )

	# Importing candles...
	candles = import_candles(ticker, timeframe, dt_start, dt_end)

	utils.merge_candles_and_trades( candles, trades )

	return( candles, trades, dt_start, dt_end, dt_train_end )


def calculate_data( candles, trades, calculate_input_fn, look_back, train_vs_test ) 

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

	utils.calculate_inputs( candles, trades, data, calculate_input, look_back )
	return data

