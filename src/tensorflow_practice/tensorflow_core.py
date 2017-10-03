# various little TensorFlow tests to demonstrate tf's features

import tensorflow as tf

# constant nodes, these output a constant value when evaluated
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

# the print statement above does not output 3.0 or 4.0 until evaluated with a session
# a Session encapsulates the control and state of the TensorFlow runtime
sess = tf.Session()
print(sess.run([node1, node2]))

# you can make more complicated computations by combining Tensor nodes with operations.
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

# graphs can be parameterized to to accept external inputs, known as placeholders. A
# placeholder is a promise to provide a value later.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# the computational graph can be made more complex by adding another operation
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# variables allow adding trainable parameters to a graph, they are constructed with
# a type and initial value. Unlike constants, variables are not initialized by the
# tf.Variable call.
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# instead, they are initialized like this. Init is a handle to the tf subgraph that
# initializes all the global variables. Until we call sess.run(), the variables are
# uninitialized
init = tf.global_variables_initializer()
sess.run(init)

# to evaluate the model, need a y placeholder to provide desired values, and need to
# write a loss function. A loss function measures how far apart apart the current
# model is from the provided data.
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# in this simple case, we can input the perfect values manually to check the loss
# function.
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
