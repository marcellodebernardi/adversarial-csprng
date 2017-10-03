# Tensors are not data structures of themselves, bur rather they are handles
# to data structures to be used when the graph is executed. That is, they
# represent a data structure, but the data held in a Tensor isn't directly
# accessible as if though from a list.
#
# Tensors need to be evaluated first, using tensor.eval(). This method only
# works if an active Session is running, that is, if the graph is being
# executed. The method returns a numpy array with the same values as in the
# Tensor. Placeholders cannot be evaluated unless a value for the placeholder
# is provided.
#
# Crucially, A TENSOR OBJECT REPRESENTS DEFERRED COMPUTATION ON A DATA
# STRUCTURE, not a concrete data structure.
# ===========================================================================

import tensorflow as tf

constant = tf.constant([1, 2, 3])
tensor = constant * constant;
print(tensor.eval())
