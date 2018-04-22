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
    """ Activation function that uses the (optional) given standard activation
        function and then applies a modulo operation to its output.

        :param divisor: tensor or int, the divisor used in computing the modulo
        :param with_activation: optional activation function to wrap
    """

    def closure(input_value: tf.Tensor) -> tf.Tensor:
        if with_activation is not None:
            input_value = with_activation(input_value)
        return tf.mod(input_value, divisor)

    return closure
