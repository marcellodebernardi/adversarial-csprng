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
    """ Activation function that uses the given standard activation
        function and then applies a modulo operation to its output. """

    def mod_act(input_value: tf.Tensor) -> tf.Tensor:
        if with_activation is not None:
            input_value = with_activation(input_value)
        return tf.mod(input_value, divisor)

    return mod_act


def bounding_clip(max_bound, negatives=False):
    """ Activation function that scales the output from the range [-max, max] to
        the range [0, 1]. Everything below -max is mapped to 0, and everything above
        max is mapped to 1. Within the range [-max, max], everything is mapped
        linearly into the [0, 1] range."""

    def activation(input_value: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(input_value, -max_bound if negatives else 0, max_bound)

    return activation


def leaky_bounding_clip(max_bound, negatives=False, alpha=0.01):
    lower_bound = tf.constant(-max_bound if negatives else 0)
    max_bound = tf.constant(max_bound)

    def activation(input_value: tf.Tensor) -> tf.Tensor:
        return tf.cond(
            tf.less(input_value, lower_bound),
            lambda: tf.add(lower_bound, tf.mul(tf.sub(input_value, lower_bound), alpha)),
            tf.cond(
                tf.greater(input_value, max_bound),
                lambda: tf.add(max_bound, tf.mul(tf.sub(input_value, max_bound), alpha)),
                lambda: input_value)
        )

    return activation
