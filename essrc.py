# Prepares source file to feed "es.py" with a list of train-test data files.
import sys
import re
from os import listdir
from os.path import isfile, join

if len(sys.argv) < 2:
	print('USE: essrc.py dir_path')
	sys.exit(0)

dir_path = sys.argv[1]

train_prefix = 'tr_' 
test_prefix = 'val_'
file_ending = '\\.csv'

output_file_name = 'es.lst'

for a in range(2, len(sys.argv)): # Reading additional parameters
	m = re.match(r'([a-zA-Z0-9\_ ]+=[0-9a-zA-Z \.\'\"\_]+)', sys.argv[a])
	if m:
		exec(m.group(1))

pattern = '^(%s)(.*%s)$' % (train_prefix,file_ending) 

counter = 0
output_file_opened = False
try:
	output_file_handle = open(output_file_name, "w")
	output_file_opened = True

	for file_name in listdir(dir_path):
		full_file_name = join( dir_path, file_name )
		if not isfile( full_file_name ):
			continue

	# Searching for files in a dir specified 
		matched = re.match( pattern, file_name, re.M|re.I)
		if not matched:
			continue

		val_file_name = test_prefix + matched.group(2)

		line = join(dir_path, file_name) + '\t' + str(counter) + '\t1\n'
		output_file_handle.write( line )

		line = join(dir_path, val_file_name) + '\t' + str(counter) + '\t2\n'
		output_file_handle.write( line )
		counter += 1
	# end of for

except IOError:
	sys.stderr.write( "Error: can\'t write file " + file_name + " or read data.\n" )
else:
	sys.stderr.write( "Train-test file pairs added: %d.\n" % (counter) )

if output_file_opened:
	output_file_handle.close()

