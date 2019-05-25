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
This module provides miscellaneous utility functions for writing
evaluation results to ASCII-encoded text files.
"""

import os
import numpy as np
from tqdm import tqdm
from utils import operations


def write_numbers_to_ascii_file(values, filename: str):
    """ Produces an ASCII output text file containing hexadecimal representations
        of each number produces by the generator.

        :param values: list or numpy array values produced by generator
        :param filename: name of file to write output into
    """
    # create directory structure to file if not exists
    if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
    # flatten and encode in hex
    values = operations.flatten(values)
    values = [format(np.uint16(i), 'x') for i in values]
    # write hex values to ASCII file
    with open(filename, 'w+') as file:
        for hex_val in tqdm(values, 'Writing to file ... '):
            file.write(str(hex_val) + "\n")


def write_to_file(values, filename: str):
    """ Writes the given data array to the given file. No prettifying.

        :param values: list or numpy array of values to log into text file
        :param filename: name of file to log into
    """
    if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

    with open(filename, 'w+') as file:
        file.write(str(operations.flatten(values)))
