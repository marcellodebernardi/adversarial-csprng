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
This module provides utility methods for obtaining values and sequences
to be used as inputs to the neural networks.
"""

import numpy as np
import tensorflow as tf


def reference_distribution_np(batch_size, seq_length, max_val) -> np.ndarray:
    """ Returns a batch of inputs sampled from the reference distribution, i.e.
    from the random uniform distribution. """
    return np.random.uniform(size=[batch_size, seq_length], low=0, high=max_val)


def reference_distribution_tf(batch_size, seq_length, max_val) -> tf.Tensor:
    """ Returns a batch of inputs sampled from the reference distribution, i.e.
    from the random uniform distribution. """
    return tf.random_uniform(shape=[batch_size, seq_length], minval=0, maxval=max_val)


def noise_prior_np(batch_size, seq_length, max_val) -> np.ndarray:
    """ Returns a batch of inputs sampled from the noise prior, i.e. to be given
    as input to a generator. """
    # todo make own implementation
    return get_input_batch_np(batch_size, max_val, False)


def noise_prior_tf(batch_size, seq_length, max_val) -> tf.Tensor:
    """ Returns a batch of inputs sampled from the noise prior, i.e. to be given
    as inputs to the generator. """
    return get_input_batch_tf(batch_size, max_val, False)


def get_input_dataset(batch_size, num_of_batches, max_val, random_offsets=False) -> tf.data.Dataset:
    """ Returns a Dataset representing an entire training dataset. """
    batches = []
    # generate batches
    for batch in range(num_of_batches):
        batches.append(get_input_batch_tf(batch_size, max_val, random_offsets))
    # stack batches into single tensor
    data = tf.stack(batches)
    # create dataset of batches
    return tf.data.Dataset.from_tensor_slices(data)


def get_input_batch_tf(batch_size, max_val, random_offsets=False) -> tf.Tensor:
    """ Returns a Tensor representing a single batch of generator
    input noise. The array is a 2D array of input samples, where each
    input sample is an array of dimensions suitable to be used as input
    for the generator network.

    In each input sample, the first value is the "random seed", which is
    fixed for the batch. The value is fixed for each batch because we do
    not want to introduce too much randomness into the generator's behavior
    using the seed. The second value in the input sample is the "counter"
    of the input, which, for training, is randomly selected.

        :param batch_size: number of input examples in each batch
        :param max_val: maximum scalar value of each tensor element
        :param random_offsets: whether offsets are fixed or randomly distributed
    """
    # fixed seed sampled from uniform distribution
    seeds = tf.fill([batch_size], tf.random_uniform(shape=[], minval=0, maxval=max_val))
    # uniformly distributed offset, or fixed offset sampled from uniform distribution
    offsets = tf.random_uniform(shape=[batch_size], minval=0, maxval=batch_size) if random_offsets \
        else tf.fill([batch_size], tf.random_uniform(shape=[], minval=0, maxval=batch_size))

    return tf.transpose(tf.stack([seeds, offsets], ))


def get_input_batch_np(batch_size, max_val, random_offsets=False) -> np.ndarray:
    """ Returns a numpy array representing a single batch of generator
    input noise. The array is a 2D array of input samples, where each
    input sample is an array of dimensions suitable to be used as input
    for the generator network.

    In each input sample, the first value is the "random seed", which is
    fixed for the batch. The value is fixed for each batch because we do
    not want to introduce too much randomness into the generator's behavior
    using the seed. The second value in the input sample is the "counter"
    of the input, which, for training, is randomly selected.

        :param batch_size: number of input examples in each batch
        :param max_val: maximum scalar value of each tensor element
        :param random_offsets: whether offsets are fixed or randomly distributed
    """
    # fixed seed sampled from uniform distribution
    seeds = np.full([batch_size], fill_value=np.random.uniform(size=[], low=0, high=max_val))
    # uniformly distributed offsets, or fixed offset sampled from uniform distribution
    offsets = np.random.uniform(size=[batch_size], low=0, high=batch_size) if random_offsets \
        else np.full([batch_size], fill_value=np.random.uniform(size=[], low=0, high=batch_size))

    return np.transpose(np.stack([seeds, offsets]))


def get_eval_input_numpy(seed, length, batch_size) -> np.ndarray:
    """ Returns an input dataset that can be used to produce a full output
        sequence using a trained generator. This method returns a 2D numpy array
        where each inner array is an (seed, offset) pair. The seed remains
        fixed, while the offset is incremented by 1 at each position.

        :param seed: pre-generated seed for the evaluation input
        :param length: number of batches in evaluation input
        :param batch_size: number of inputs in each batch
    """
    data = []
    offset = 0

    for batch_num in range(length):
        batch = []
        for item in range(batch_size):
            batch.append([seed, offset])
            offset = offset + 1
        data.append(batch)

    return np.array(data)
