# License: See LICENSE
# Fit a straight line, of the form y=m*x+b

import tensorflow as tf
import numpy as np

'''
Your dataset.
'''
xs = np.linspace(0.0, 8.0, 8000000) # 8-million features
ys = 0.3*xs-0.8+np.random.normal(scale=0.25, size=len(xs)) # 8-million labels

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
Define placeholders for big data.
'''
_BATCH = 8 # Use only eight points at a time.
xs_placeholder = tf.placeholder(tf.float32, [_BATCH])
ys_placeholder = tf.placeholder(tf.float32, [_BATCH]) 

'''
Define the error between the data and the model as a tensor (distributed computing).
'''
ys_model = m*xs_placeholder+b # Tensorflow knows this is a vector operation
total_error = tf.reduce_sum((ys_placeholder-ys_model)**2) # Sum up every item in the vector

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

	_EPOCHS = 10000 # Number of "sweeps" across data
	for iteration in range(_EPOCHS):
		random_indices = np.random.randint(len(xs), size=_BATCH) # Randomly sample the data
		feed = {
			xs_placeholder: xs[random_indices],
			ys_placeholder: ys[random_indices]
		}
		session.run(optimizer_operation, feed_dict=feed) # Call operator

	slope, intercept = session.run((m, b)) # Call "m" and "b", which are operators
	print('Slope:', slope, 'Intercept:', intercept)

