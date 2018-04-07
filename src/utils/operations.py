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
This module provides utility functions for performing operations on
the data processed by the models, such as flattening of lists with
arbitrary nested structure, n-base element-wise logarithms on tensors,
and more.
"""

import tensorflow as tf
import numpy as np
from keras import Model


def slice_gen_out(generator_output: np.ndarray) -> (np.ndarray, np.ndarray):
    """ For an array of outputs produced by a generator, where each element in the
        array is an array of real numbers, splits all the inner array into two, such that
        the first resulting array contains all elements of the original inner array minus
        n_to_predict items, and the second contains the last n_to_predict items.
    """
    data = generator_output
    return data[:, :-1], np.reshape(data[:, -1], [len(generator_output), 1])


def log(x, base) -> tf.Tensor:
    """ Allows computing element-wise logarithms on a Tensor, in
        any base. TensorFlow itself only has a natural logarithm
        operation.
    """
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator


def flatten(weight_matrix) -> list:
    """ Allows flattening a matrix of iterables where the specific type
        and shape of each iterable is not necessarily the same. Returns
        the individual elements of the original nested iterable in a single
        flat list.

        Note that this operation's implementation is rather slow, and should
        be avoided unless necessary.
    """
    flattened_list = []
    try:
        for element in weight_matrix:
            flattened_list.extend(flatten(element))
        return flattened_list
    except TypeError:
        return weight_matrix
