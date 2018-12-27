import sys, re
import keras



def create_nn( num_features, num_classes=2, optimizer=None, activation=None, units_multiplier=None, num_hidden_layers=None ):
	if isinstance(num_features,str):
		return( True ) # Returns "True" if the model requires one-hot encoding and "False" otherwise]

	if units_multiplier is None:
		units_multiplier = 1
	if optimizer is None:
		optimizer = 'adam'
	if activation is None:
		activation = 'tanh'
	if num_hidden_layers is None:
		num_hidden_layers = 1

	m = Sequential()
	m.add(Dense(units=num_features*units_multiplier, activation=activation, input_dim=num_features, kernel_initializer=he_uniform(1)))
	for l in range(num_hidden_layers):
		m.add(Dense(num_features*units_multiplier, activation=activation))
	m.add(Dense(num_classes, activation='softmax'))
	# Compiling the model
	#m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	m.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])	
	return m
# end of create_model()



