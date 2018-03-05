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


def set_trainable(model: Model, optimizer, loss, recompile, trainable: bool = True):
    """Helper method that sets the trainability of all of a model's
    parameters."""
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
    if recompile:
        model.compile(optimizer, loss)


def split_n_last(generator_output: np.ndarray, n_to_predict=1) -> (np.ndarray, np.ndarray):
    """For an array of outputs produced by a generator, where each element in the
    array is an array of real numbers, splits all the inner array into two, such that
    the first resulting array contains all elements of the original inner array minus
    n_to_predict items, and the second contains the last n_to_predict items."""
    inputs = []
    outputs = []

    for i in range(len(generator_output)):
        inp, out = split_generator_output(generator_output[i], n_to_predict)
        inputs.append(inp)
        outputs.append(out)

    return np.array(inputs), np.array(outputs)


def split_generator_output(generator_output: np.ndarray, n_to_predict) -> (np.ndarray, np.ndarray):
    """Takes the generator output as a numpy array and splits it into two
    separate numpy arrays, the first representing the input to the predictor
    and the second representing the output labels for the predictor."""
    seq_len = len(generator_output)
    predictor_inputs = generator_output[0: -n_to_predict]
    predictor_outputs = generator_output[seq_len - n_to_predict - 1: seq_len - n_to_predict]
    return predictor_inputs, predictor_outputs


def get_ith_batch(data: np.ndarray, batch: int, batch_size):
    """Returns a slice of the given array, corresponding to the ith
    batch of size batch_size."""
    return data[(batch * batch_size): (batch + 1) * batch_size]


def log(x, base) -> tf.Tensor:
    """Allows computing element-wise logarithms on a Tensor, in
    any base. TensorFlow itself only has a natural logarithm
    operation."""
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator


def flatten_irregular_nested_iterable(weight_matrix) -> list:
    """Allows flattening a matrix of iterables where the specific type
    and shape of each iterable is not necessarily the same. Returns
    the individual elements of the original nested iterable in a single
    flat list.
    """
    flattened_list = []
    try:
        for element in weight_matrix:
            flattened_list.extend(flatten_irregular_nested_iterable(element))
        return flattened_list
    except TypeError:
        return weight_matrix
