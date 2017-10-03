# Variables encapsulate a Tensor and are persistent across Sessions.
# They can be grouped into Collections and placed on specific devices.
#
# Before a variable is used, it must be initialized. In the lowest-level
# TensorFlow API (TensorFlow core), initialization must be done
# explicitly. The high-level frameworks initialize variables
# automatically.
# ======================================================================

import tensorflow as tf

some_variable = tf.Variable()
session = tf.Session()

# To initialize all trainable variables in one go, before training starts
session.run(tf.global_variables_initializer())

# You can also initialize a variable individually
session.run(some_variable.initializer())

# You can check for uninitialized variables
print(session.run(tf.report_uninitialized_variables()))

# Variables can also be shared, in one of two ways:
#   1. explicitly pass it around
#   2. implicitly wrapping it into a tf.variable_scope object
