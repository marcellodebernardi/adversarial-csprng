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
This module provides utility methods for obtaining values and sequences
to be used as inputs to the neural networks.
"""

import numpy as np
import tensorflow as tf


def get_input_tensor(batch_size, max_val) -> tf.Tensor:
    """ Returns a symbolic Tensor operation for sampling inputs to the
     generator in the GAN. The specification of the inputs is the same
     as for get_inputs_as_array. """
    return tf.transpose(
        tf.stack(
            [tf.fill([batch_size], tf.random_uniform(shape=[], minval=0, maxval=max_val)),
             tf.random_uniform(shape=[batch_size], minval=0, maxval=max_val)],
        ))


def get_input_numpy(batch_size, max_val) -> np.ndarray:
    """ Returns a symbolic Tensor operation for sampling inputs to the
     generator in the GAN. The specification of the inputs is the same
     as for get_inputs_as_array. """
    return np.transpose(
        np.stack(
            [np.full([batch_size], fill_value=np.random.uniform(size=[], low=0, high=max_val)),
             np.random.uniform(size=[batch_size], low=0, high=max_val)]
        ))


def get_eval_input_numpy(seed, length, batch_size) -> np.ndarray:
    """ Returns an input dataset that can be used to produce a full output
    sequence using a trained generator. This method returns a 2D numpy array
    where each inner array is an (seed, offset) pair. """
    data = []
    offset = 0

    for batch_num in range(length):
        batch = []
        for item in range(batch_size):
            batch.append([seed, offset])
            offset += 1
        data.append(batch)

    return np.array(data)
