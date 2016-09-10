#  Fit a straight line, of the form y=m*x+b

import tensorflow as tf

xs = [ 0.00,  1.00,  2.00, 3.00, 4.00, 5.00, 6.00, 7.00] # Features
ys = [-0.82, -0.94, -0.12, 0.26, 0.39, 0.64, 1.02, 1.00] # Labels

'''
with enough iterations, initial weights dont matter since our cost function is convex.
'''
m_initial = -0.5 # Initial guesses
b_initial =  1.0


'''
define free variables to be solved. we'll be taking partial derivatives of m and b with respect to j (cost).
'''
m = tf.Variable(m_initial) # Parameters
b = tf.Variable(b_initial)

error = 0.0
for i in range(len(xs)):
	y_model = m*xs[i]+b # Output of the model aka yhat
	error += (ys[i]-y_model)**2 # Difference squared - this is the "cost" to be minimized


'''
once cost function is defined, use gradient descent to find global minimum.
'''

operation = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(error) # Does one step



with tf.Session() as session:
	session.run(tf.initialize_all_variables()) # Initialize session

	_EPOCHS = 10000 # number of "sweeps" across data

	for iteration in range(_EPOCHS):
		session.run(operation)

	print('Slope:', m.eval(), 'Intercept:', b.eval())

