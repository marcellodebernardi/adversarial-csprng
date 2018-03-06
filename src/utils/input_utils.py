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
from tqdm import tqdm


def get_seed_dataset(size, max_val) -> (np.ndarray, np.ndarray):
    """Generates a dataset for adversarially training Jerry (the generator).
    The first element in the returned tuple is an array of seeds, representing
    the inputs to the GAN. The second element is an array of labels, which are
    all 0 as the correct label for generated sequences, as labelled by the
    discriminator, is 0."""
    x = []
    # generate dataset
    for element in tqdm(range(size), 'Obtaining seeds'):
        x.append(get_random_value(max_val))
    return np.array(x), np.zeros(len(x))


def get_sequences_dataset(generator: Model, seeds, sequence_length, max_val) -> (np.ndarray, np.ndarray):
    """Generates a dataset for training Diego (the discriminator). The
    dataset consists half of truly random sequences and half of sequences
    produced by the generator."""
    dataset = generator.predict(seeds)
    labels = [0 for i in range(len(seeds))]

    for i in range(int(len(seeds)/2)):
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
    if source == 'random':
        return np.array([rng.uniform(0, max_val) for i in range(int(sequence_length))], dtype=np.float64)
    elif source == 'system_random':
        return np.array([rng.SystemRandom().uniform(0, max_val) for i in range(int(sequence_length))], dtype=np.float64)


def get_random_value(max_val, source='random') -> float:
    """Returns a floating point number between 0 and max_val, obtained from
    the specific source of randomness."""
    if source == 'random':
        return rng.uniform(0, max_val)
    elif source == 'system_random':
        return rng.SystemRandom().uniform(0, max_val)
