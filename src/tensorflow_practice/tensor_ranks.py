# The code snippets in this file demonstrate creating tensorflow
# variables of different ranks. Recall that in tensorflow tensors
# are just a generalization of vectors and matrices, in that they
# are n-dimensional data structures. So a rank-0 tensor is a
# scalar value, rank-1 is a vector, rank-2 a matrix, rank-3 is a
# 3-tensor, and so on.
# ===============================================================

import tensorflow as tf

# rank 0 tensors
mammal = tf.Variable('Elephant', tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.1415, tf.float64)
its_complicated = tf.Variable((12.3, -4.85), tf.complex64)

# rank 1 tensors
mystr = tf.Variable(['Hello'], tf.string)
cool_numbers = tf.Variable([3.1415, 2.71828], tf.float32)
first_primes = tf.Variable([1, 2, 3, 4, 5], tf.int16)
its_very_complicated = tf.Variable([(1, -3), (2, -4)], tf.complex64)

# rank 2 tensors
squares = tf.Variable([[1, 2, 4], [1, 5, 6]], tf.int32)

# to determine the rank of a tensor, use tf.rank
my_rank = tf.rank(squares)
print(my_rank)

# you can also slice tensors as you would do with lists; slicing can
# access not only scalars in the tensor but subvectors, submatrices
# and subtensors
