from operations import round_tensor

"""
The activations.py module defines activation functions that can
be used in Keras layers. Activation functions operate on scalar
values and thus do not need to be implemented as symbolic TensorFlow
operations.
"""


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


def to_integer(activation_function):
    """Produces a wrapper for another activation function, which
    reduces the value produced by that activation to an integer."""
    def act(input_value):
        return round_tensor(activation_function(input_value))
    return act
