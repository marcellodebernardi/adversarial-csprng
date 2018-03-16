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
import random as rng
from keras import Model


def get_eval_input(seed, length):
    """ Returns an input dataset that can be used to produce a full output
    sequence using a trained generator. This method returns a 2D numpy array
    where each inner array is an (seed, offset) pair. """
    return np.array([[seed, offset] for offset in range(length)])


def get_inputs(size, max_val) -> np.ndarray:
    """Generates an input dataset for adversarially training the generator. The
    dataset is structured as a numpy array of dimensions (2, size). For array[0],
    each element of the inner array is """
    seed = get_random_value(max_val)
    initial_offset = int(get_random_value(size))
    x = [[], [], []]
    # generate dataset
    for element in range(size):
        x[0].append(seed)
        x[1].append(element + initial_offset)
        x[2].append(0)

    return np.array(x)


def get_sequences(generator: Model, inputs: np.ndarray, sequence_length, max_val) -> (np.ndarray, np.ndarray):
    """Generates a dataset for training Diego (the discriminator). The
    dataset consists half of truly random sequences and half of sequences
    produced by the generator."""
    dataset = generator.predict_on_batch(inputs).tolist()
    labels = [0 for i in range(len(inputs))]

    for i in range(int(len(inputs))):
        dataset.append(get_random_sequence(sequence_length, max_val))
        labels.append(1)

    combined = list(zip(dataset, labels))
    rng.shuffle(combined)
    dataset[:], labels[:] = zip(*combined)
    return np.array(dataset), np.array(labels)


def get_random_sequence(sequence_length, max_val, source='random') -> np.ndarray:
    """Returns a numpy array of given length containing uniformly distributed
    real numbers in the range [0, max_val]. Such a random sequence is appropriate
    to be used as an input to a discriminator / predictor.

    :parameter max_val: the highest value that each individual real number can have
    :parameter sequence_length: length of each individual sequence of reals
    :parameter source: allows specification of the source of randomness
    """
    return np.array([get_random_value(max_val, source) for i in range(int(sequence_length))], dtype=np.float64)


def get_random_value(max_val, source='random') -> float:
    """Returns a floating point number between 0 and max_val, obtained from
    the specific source of randomness."""
    if source == 'random':
        return rng.uniform(0, max_val)
    elif source == 'system_random':
        return rng.SystemRandom().uniform(0, max_val)
