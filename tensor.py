# License: See LICENSE
# Fit a straight line, of the form y=m*x+b

import tensorflow as tf

'''
Your dataset.
'''
xs = [ 0.00,  1.00,  2.00, 3.00, 4.00, 5.00, 6.00, 7.00] # Features
ys = [-0.82, -0.94, -0.12, 0.26, 0.39, 0.64, 1.02, 1.00] # Labels

'''
Initial guesses, which will be refined by TensorFlow.
'''
m_initial = -0.5 # Initial guesses
b_initial =  1.0

'''
Define free variables to be solved.
'''
m = tf.Variable(m_initial) # Parameters
b = tf.Variable(b_initial)

'''
Define the error between the data and the model as a tensor (distributed computing).
'''
ys_model = m*xs+b # Tensorflow knows this is a vector operation
total_error = tf.reduce_sum((ys-ys_model)**2) # Sum up every item in the vector

'''
Once cost function is defined, create gradient descent optimizer.
'''
optimizer_operation = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(total_error) # Does one step

'''
Create operator for initialization.
'''
initializer_operation = tf.global_variables_initializer()

'''
All calculations are done in a session.
'''
with tf.Session() as session:

	session.run(initializer_operation) # Call operator

	_EPOCHS = 10000 # number of "sweeps" across data
	for iteration in range(_EPOCHS):
		session.run(optimizer_operation) # Call operator

	slope, intercept = session.run((m, b)) # Call "m" and "b", which are operators
	print('Slope:', slope, 'Intercept:', intercept)

