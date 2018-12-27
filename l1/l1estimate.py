import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import re
import numpy as np
import l1models
from l2utils import read_signals

FILE_NAME = 'test.txt'
CALC_OUTPUT_FN = l1models.calculate_output_mean2
THRESHOLD = 0.1

command_line_arguments = ''
for a in range(1, len(sys.argv)): # Reading additional parameters
    m = re.match(r'([a-zA-Z0-9\_ ]+=[0-9a-zA-Z \.\'\"\_\)\(\,)]+)', sys.argv[a])
    if m:
        exec(m.group(1))
    command_line_arguments += ' ' + sys.argv[a]

lookforward, num_classes = CALC_OUTPUT_FN('query_lookforward')

signals = read_signals( FILE_NAME )

num_good = 0
num_bad = 0
error = 0.0
for s in range(len(signals) - lookforward):
	predicted = [0]*num_classes
	for c in range(num_classes):
		predicted[c] = signals[s]['output'+str(c)]
	argmax = np.argmax(predicted)

	if argmax == 0 or argmax == num_classes-1:
		if predicted[argmax] > 1.0 / num_classes + THRESHOLD:
			actual = signals[s+lookforward]['actual_output']
			error += np.abs(argmax - actual)
			if actual == argmax:
				num_good += 1
			else:
				num_bad += 1

num_good_plus_num_bad = float(num_good + num_bad)
if num_good_plus_num_bad > 0.0:
	print( 'threshold=%f, num_good=%d, num_bad=%d, guess rate=%f, error=%f' % \
		(THRESHOLD, num_good, num_bad, float(num_good) / float(num_good+num_bad), error / float(num_good+num_bad) ) )
else:
	print('No trades are possible with the chosen threshold %s' %(THRESHOLD))