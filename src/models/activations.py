"""
The activations.py module defines activation functions that can
be used in Keras layers. Activation functions operate on scalar
values and thus do not need to be implemented as symbolic TensorFlow
operations.
"""

import tensorflow as tf


def modulo(divisor, activation_function=None):
    """Activation function that uses the given standard activation
        function and then applies a modulo operation to its output."""
    def mod_act(input_value):
        if activation_function is not None:
            return activation_function(input_value) % divisor
        else:
            return input_value % divisor
    return mod_act


def absolute(input_value):
    """Returns the absolute value of the given input."""
    return abs(input_value)


def diagonal_max(max_bound):
    """Activation function that scales"""
    def activation(input_value):
        input_value = tf.clip_by_value(input_value, -max_bound, max_bound)
        return tf.div(tf.add(input_value, tf.constant(max_bound, dtype=tf.float32)), tf.constant(max_bound * 2, dtype=tf.float32))
    return activation
