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
Module provides wrappers around various sources of randomness in
order to generate the sort of data used by the networks in this
project.
"""

import numpy as np
import random as rng
import quantumrandom as qr


def get_seed_dataset(max_seed: int, seed_size: int, unique_seeds: int, repetitions: int, batch_size=1) -> np.ndarray:
    """Returns a seed dataset for training. Each individual seed consists of
    n = seed_size real numbers in the range [0 - max_seed]. The dataset contains
    k = (unique_seeds * repetitions) seeds, split into batches of size batch_size.
    The default batch size of 1 results in a dataset suitable for online training.
    """
    # todo
    # check for bad input
    if (unique_seeds * repetitions) % batch_size != 0:
        raise ValueError('The product (unique_seeds * repetitions) must be a multiple of the batch size')
    # generate unique seeds
    seeds = [[rng.uniform(0, max_seed) for i in range(seed_size)] for j in range(unique_seeds)]
    # expand to include repetition of unique seeds
    seeds = np.array([seed for seed in seeds for i in range(repetitions)], dtype=np.float64)
    # split into batches
    if unique_seeds == 1 and repetitions == 1:
        return np.expand_dims(seeds, 0)
    else:
        return np.array(np.split(seeds, int(len(seeds) / batch_size)), dtype=np.float64)


def get_random_sequence(max_val, length, seq_num, backend='random') -> np.ndarray:
    """Returns a matrix of random real numbers, such that the first dimension
    identifies a particular number sequence, and the second dimension identifies
    a value within a sequence. Each sequence is an input to the discriminator.
    """
    if backend == 'random':
        return np.array([[rng.uniform(0, max_val) for i in range(length)] for j in range(seq_num)], dtype=np.float64)
    elif backend == 'system_random':
        return np.array([[rng.SystemRandom().uniform(0, max_val) for i in range(length)] for j in range(seq_num)], dtype=np.float64)
    elif backend == 'quantum_random':
        return np.array([[float.fromhex(val) for val in qr.hex(length)] for j in range(seq_num)], dtype=np.float64)

