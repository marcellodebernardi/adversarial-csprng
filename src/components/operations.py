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
This module defines operations used in the generative adversarial network
implementations.
"""

import numpy as np
import tensorflow as tf


def slice_gen_out_tf(generator_output: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """ For an array of outputs produced by a generator, where each element in the
        array is an array of real numbers, splits all the inner array into two, such that
        the first resulting array contains all elements of the original inner array minus
        n_to_predict items, and the second contains the last n_to_predict items.

        :param generator_output: numpy array holding the generator's output vector
    """
    data = generator_output
    return data[:, :-1], np.reshape(data[:, -1], [tf.shape(generator_output)[0], 1])


def combine_generated_and_reference_tf(generated: tf.Tensor, reference: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """ Concatenates a Tensor containing samples produced by the generator and
    a Tensor containing samples obtained from the reference distribution, and
    returns a tuple consisting of the concatenated tensors, and the corresponding
    labels.

    That is, for two tensors [gen, gen, gen] and [ref, ref, ref], this function
    returns ([gen, gen, gen, ref, ref, ref], [0, 0, 0, 1, 1, 1]). """
    data = tf.concat([generated, reference], 0)
    labels = tf.concat([tf.zeros([tf.shape(generated)[0]]), tf.ones(tf.shape(reference)[0])], 0)

    return data, labels
