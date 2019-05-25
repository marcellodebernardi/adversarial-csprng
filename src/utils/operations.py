# Marcello De Bernardi, University of Oxford
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
This module provides utility functions for performing operations on
the data processed by the models, such as flattening of lists with
arbitrary nested structure, n-base element-wise logarithms on tensors,
and more.
"""

import tensorflow as tf


def log(x, base) -> tf.Tensor:
    """ Allows computing element-wise logarithms on a Tensor, in
        any base. TensorFlow itself only has a natural logarithm
        operation.

        :param x: tensor to compute element-wise logarithm for
        :param base: base to compute logarithm in
    """
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator


def flatten(irregular_matrix) -> list:
    """ Allows flattening a matrix of nested iterables where the specific type
        and shape of each iterable is not necessarily the same. Returns
        the individual elements of the original nested iterable in a single
        flat list.

        Note that this operation's implementation is rather slow, and should
        be avoided unless necessary.

        :param irregular_matrix: a nested iterable with irregular dimensions
    """
    flattened_list = []
    try:
        for element in irregular_matrix:
            flattened_list.extend(flatten(element))
        return flattened_list
    except TypeError:
        return irregular_matrix
