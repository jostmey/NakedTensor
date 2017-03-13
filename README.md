# NakedTensor

This is a bare bottom example of machine learning in TensorFlow. You will not find a simplier example for introducing TensorFlow.

In each example, a straight line is fit to some data. The slope and y-intercept of the line are determined using gradient descent. If you do not know what is gradient descent, check out the wikipedia page [link](https://en.wikipedia.org/wiki/Gradient_descent).

After creating the required variables, the error between the data and the line is *defined*. The definition of the error is then plugged into the optimizer. Once TensorFlow has been started, the optimizer is repeatedly called to iteratively fit the line to the data.

Read the scripts in this order:
 * serial.py
 * tensor.py

## serial.py

The purpose of this script is to illustrate the nuts and bolts of a TensorFlow model. The script makes it easy to understand how model is put together. The error between the data and the line is defined using a for loop. Because of the way the error is defined, the caclulation runs in serial.

## tensor.py

This script goes a step farther than `serial.py` although it actually requires fewer lines of code. The outline of the code is the same as before except this time the error is defined using tensor operations. Because tensors are used, the code can run in parallel.

When you use tensors, the error at each point of data can be run on a separate computing core. There are 8 points of data, so if you have a computer with eight cores it should run almost eight times faster.

## Thanks

That's it. Hopefully you found this tutorial enlightning.

