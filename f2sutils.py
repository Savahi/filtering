import sys
import numpy as np
from futils import calculate_pnl

def test_models( candles, trades, data, models, num_classes, model_type='nn',
	filter_by_worst_class=False, print_header=False, prediction_threshold=1.0001, side=0 ):

	file = sys.argv[1] 
	command_line_arguments = ''
	for a in range(2, len(sys.argv)): # Reading additional parameters
		m = re.match(r'([a-zA-Z0-9\_ ]+=[0-9a-zA-Z \.\'\"\_]+)', sys.argv[a])
		if m:
			exec(m.group(1))
		command_line_arguments += ' ' + sys.argv[a]

	num_models = len(models)
	num_trades = 0
	num_trades_oked = 0
	oked_right = 0
	oked_wrong = 0
	profit_actual = 0
	profit_optimized = 0
	trades_actual = []
	trades_oked = []

	for t in range(len(data['test_inputs'])):
		inp = np.array( [ data['test_inputs'][t] ] )
		#inp = scaler.transform(inp)		
		trade_num = data['test_trade_num'][t]
		profit = trades[trade_num]['profit']

		if not filter_by_worst_class:
			filter_class = num_classes-1
		else:
			filter_class = 0
		num_models_that_confirmed_filter_class = 0

		for m in range(num_models): # Iterating through all the models trained to obtain predictions
			prediction = models[m].predict_proba( inp )[0]
			sys.stderr.write(str(prediction) + "\n")
			sys.stderr.write(str(data['test_outputs'][t]) + "\n")
			if np.argmax(prediction) == filter_class: 
				if prediction[filter_class] > prediction_threshold * 1.0/num_classes:
					num_models_that_confirmed_filter_class += 1
		
		profit_actual += profit
		num_trades += 1

		trades_actual.append(trades[trade_num])

		oked = False
		if not filter_by_worst_class: # Filtering by best class
			if num_models_that_confirmed_filter_class == num_models:
				oked = True
		else: # Filtering by worst class
			if num_models_that_confirmed_filter_class < num_models:
				oked = True

		if oked: # If oked...
			trades_oked.append(trades[trade_num])
			profit_optimized += profit
			num_trades_oked += 1			
			if model_type == 'nn':
				right = data['test_outputs'][t][filter_class] == 1
			else:
				right = data['test_outputs'][t] == filter_class

			if right:
				oked_right += 1
			else:
				oked_wrong += 1

		sys.stderr.write("Trades (actual/oked): %d (%d), oked right: %d, oked wrong: %d, profit_actual=%f, profit_optimized=%f\n" % \
			(num_trades, num_trades_oked, oked_right, oked_wrong, profit_actual, profit_optimized))

	if print_header:
		print("File, Side, Num. of Trades, Num. of Trades Allowed, +, -, Profit, Optimized Profit, CMA")
	print("%s, %d, %d, %d, %d, %d, %f, %f %s" % \
		(file, side, num_trades, num_trades_oked, oked_right, oked_wrong, profit_actual, profit_optimized, command_line_arguments))

	'''
	pnl = {}
	calculate_pnl(candles, trades_actual, pnl, "trades_actual_")
	calculate_pnl(candles, trades_oked, pnl, "trades_oked_")

	import matplotlib.pyplot as plt
	plt.plot( pnl['time_dt'], pnl['trades_actual_pnl'] )
	plt.plot( pnl['time_dt'], pnl['trades_oked_pnl'] )
	plt.savefig( file + '.png' )
	'''