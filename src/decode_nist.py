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
This command-line utility works on ASCII text files containing one base-16
numeric value per line, and converts it to a new file containing ASCII 0 and
1 characters. This output file is formatted for analysis by the NIST test
suite.
"""


import sys
import glob
from tqdm import tqdm


def main():
    """ Application entry point. """
    for filename in tqdm(glob.glob(sys.argv[1]), 'decoding (dieharder): '):
        new_filename = filename[:-4] + '_dieharder.txt'

        # for each file, open and create edited
        with open(filename) as file, open(new_filename, 'w') as new_file:
            data = file.readlines()

            for number in data:
                new_file.write(format(int(number, 16), '016b'))


if __name__ == '__main__':
    main()
