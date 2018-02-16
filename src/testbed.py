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
This module allows loading of a generator network from an .h5
file. Its purpose is to allow testing and further evaluation of a
trained generator network.
"""

import sys
from keras.models import load_model
from utils import utils
from main import MAX_VAL, VAL_BITS


def main():
    if len(sys.argv) != 1:
        print('A file path to the generator model to load was not found.')
        return

    generator = load_model(sys.argv[0])
    utils.generate_output_file(, MAX_VAL, VAL_BITS, )


if __name__ == '__main__':
    main()
