
def run( start_date, end_date, model=None, ticker, timeframe, on_bar_function, file_name ):

    file_opened = False

    try:
        file_handle = open(file_name, "w")
        file_opened = True

        if model is None:
        	model = prepare_model() # Creating an ML-model.
        alg = TNG(start_date, end_date) # Creating an instance of environment to run algorithm in.
        alg.addInstrument(ticker) # Adding an instrument.
        alg.addTimeframe(ticker, timeframe) # Adding a time frame. 
        alg.run_backtest(on_bar_function) # Backtesting...

                signals.append( new_signal )
            num_lines_read += 1
    except IOError:
        sys.stderr.write( "Error: can\'t find file " + file_name + " or read data.\n" )
    else:
        sys.stderr.write( "Lines read = %d, lines skipped = %d.\n" % (num_lines_read, num_lines_skipped) )

    if file_opened:
        file_handle.close()

