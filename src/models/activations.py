# Marcello De Bernardi, Queen Mary University of London
#
# An exploratory proof-of-concept implementation of a CSPRNG
# (cryptographically secure pseudorandom number generator) using
# adversarially trained neural networks. The work was inspired by
# the findings of Abadi & Andersen, outlined in the paper
# "Learning to Protect Communications with Adversarial
# Networks", available at https://arxiv.org/abs/1610.06918.
#
# The original implementation by Abadi is available at
# https://github.com/tensorflow/models/tree/master/research/adversarial_crypto
# =================================================================

"""
The activations.py module defines activation functions that can
be used in Keras layers. Activation functions operate on scalar
values and thus do not need to be implemented as symbolic TensorFlow
operations.
"""

import tensorflow as tf


def modulo(divisor, with_activation=None):
    """Activation function that uses the given standard activation
        function and then applies a modulo operation to its output."""
    def mod_act(input_value):
        if with_activation is not None:
            return with_activation(input_value) % divisor
        else:
            return input_value % divisor
    return mod_act


def absolute(input_value):
    """Returns the absolute value of the given input."""
    return abs(input_value)


def bounding_clip(max_bound):
    """Activation function that scales the output from the range [-max, max] to
     the range [0, 1]. Everything below -max is mapped to 0, and everything above
     max is mapped to 1. Within the range [-max, max], everything is mapped
     linearly into the [0, 1] range."""
    def activation(input_value):
        return tf.clip_by_value(input_value, -max_bound, max_bound)
    return activation


def leaky_bounding_clip(max_bound, alpha):
    """ A 'leaky' version of the diagonal bounding box activation, which has a
    small gradient alpha for all outputs outside the bounding box. """
    def activation(input_value):
        # value was outside box
        if input_value < -max_bound:
            return -max_bound + (alpha * (input_value + max_bound))
        # value was inside box
        elif input_value > max_bound:
            return max_bound + (alpha * (input_value - max_bound))
        else:
            return input_value
    return activation
