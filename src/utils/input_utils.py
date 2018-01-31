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


def get_jerry_training_dataset(batch_size, num_of_batches, unique_seeds_per_batch, max_val) -> (np.ndarray, np.ndarray):
    """Generates a dataset for adversarially training Jerry (the generator).
    The first element in the returned tuple is an array of seeds, representing
    the inputs to the GAN. The second element is an array of labels, which are
    all 0 as the correct label for generated sequences, as labelled by the
    discriminator, is 0."""
    # check for bad input
    if batch_size % unique_seeds_per_batch != 0:
        raise ValueError('The number of unique seeds must be a factor of the batch size.')
    # how many times to repeat each seed in each batch
    repetitions = batch_size / unique_seeds_per_batch
    x = []
    # generate dataset
    for batch in range(num_of_batches):
        for unique_seed in range(unique_seeds_per_batch):
            seed = get_random_value(max_val)
            for i in range(int(repetitions)):
                x.append(seed)
    return np.array(x), np.zeros(len(x))


def get_discriminator_training_dataset(generator: Model, batch_size, num_of_batches, sequence_length, max_val, shuffle=False) -> (np.ndarray, np.ndarray):
    """Generates a dataset for training Diego (the discriminator). The
    dataset consists half of truly random sequences and half of sequences
    produced by the generator."""
    dataset = []
    labels = []
    for i in range(num_of_batches):
        dataset.extend([get_random_sequence(sequence_length, 1) for i in range(int(batch_size / 2))])
        dataset.extend(generator.predict(get_random_sequence(batch_size / 2, max_val)))
        labels.extend([1 for i in range(int(batch_size / 2))])
        labels.extend([0 for i in range(int(batch_size / 2))])
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
