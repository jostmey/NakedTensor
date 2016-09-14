from tensorflow.contrib import learn
from sklearn import cross_validation
import tensorflow as tf
import numpy as np




_NUM_FEATURES = 4
_NUM_CLASSES = 3

# use iris data
iris = learn.datasets.load_dataset('iris')
X = iris.data
y = np.eye(_NUM_CLASSES)[iris.target] # 1-hot encoding since this is classification

x_train, x_test, y_train, y_test = cross_validation.train_test_split(
      X, y, test_size=0.2, random_state=42)



inp = tf.placeholder(tf.float32, [None, _NUM_FEATURES])

# w and b are weights to be learned
w = tf.Variable(tf.zeros([_NUM_FEATURES, _NUM_CLASSES]))
b = tf.Variable(tf.zeros([_NUM_CLASSES]))

# define activation function - simple matrix multiply plus translation
y = tf.nn.softmax(tf.matmul(inp, w) + b)
y_ = tf.placeholder(tf.float32, [None, _NUM_CLASSES]) # y_ refers to prediction; 1-hot encoding means prediction is a 3-unit vector
cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # cost function

# use adam for minimize cross entropy cost function
operation = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

# unlike regression, cost function cannot be used to evaluate final model; want accuracy instead
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



_EPOCHS = 100 # number of "sweeps" across data

# with a defined model, we can train & test
with tf.Session() as session:

    session.run(tf.initialize_all_variables())
     
    # train
    for iteration in range(_EPOCHS):
		session.run(operation, feed_dict={inp: x_train, y_: y_train})
    
    # test 
    print session.run(accuracy, feed_dict={inp: x_test, y_: y_test})
