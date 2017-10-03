# The shape of a Tensor is the number of elements in each dimension.
# The TensorFlow documentation uses three notational conventions to
# describe Tensor dimensionality: rank, shape, and dimension number.
# Ranks are explain in tensor_ranks.py.
#
# rank          shape           dimension number
# 0             []              0-D
# 1             [D0]            1-D
# 2             [D0, D1]        2-D
# 3             [D0, D1, D2]    3-D
# ...           ...             ...
#
# The shape of a Tensor can be accessed as a TensorShape object, via
# the tf.shape property.
# ==================================================================

import tensorflow as tf

# making a vector with as many 0s as there are columns in a given matrix
matrix = tf.Variable([[1, 2, 3], [4, 5, 6]], tf.int32)
zeros = tf.zeros(tf.shape(matrix)[1])

# the number of elements in a tensor is the product of all its
# dimensions. You can change the shape of a tensor while keeping
# its number of elements intact with the reshape() operation.

# all elements in a tensor are of the same data type. The data type
# of a tensor can be changed using tf.cast()
