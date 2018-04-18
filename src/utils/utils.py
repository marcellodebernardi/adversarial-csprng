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
This module provides miscellaneous utility functions for the training
and evaluation of the model, such as means for storing ASCII bit sequences
into text files and emailing a report after training.
"""

import os
import numpy as np
from tqdm import tqdm
from utils import operations


def write_output_file(values, filename=None):
    """
    Produces an ASCII output text file containing hexadecimal representations
    of each number produces by the generator.
    """
    if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

    values = operations.flatten(values)
    values = [format(np.uint16(i), 'x') for i in values]

    with open(filename, 'w+') as file:
        for hex_val in tqdm(values, 'Writing to file ... '):
            file.write(str(hex_val) + "\n")


def write_to_file(values, filename: str):
    """ Writes the given data array to the given file. No prettifying. """
    if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

    with open(filename, 'w+') as file:
        file.write(str(operations.flatten(values)))
