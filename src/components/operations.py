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
This module defines operations used in the generative adversarial network
implementations.
"""

import numpy as np


def slice_gen_out(generator_output: np.ndarray) -> (np.ndarray, np.ndarray):
    """ For an array of outputs produced by a generator, where each element in the
        array is an array of real numbers, splits all the inner array into two, such that
        the first resulting array contains all elements of the original inner array minus
        n_to_predict items, and the second contains the last n_to_predict items.

        :param generator_output: numpy array holding the generator's output vector
    """
    data = generator_output
    return data[:, :-1], np.reshape(data[:, -1], [len(generator_output), 1])
