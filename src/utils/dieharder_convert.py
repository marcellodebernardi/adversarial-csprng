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
Converts a binary ASCII file as outputted by the DISCGAN and PREDGAN
generators into the format required by the dieharder test suite.
"""

import sys
import glob
from tqdm import tqdm


def main():
    """ Constructs the neural networks, trains them, and logs
    all relevant information."""
    for filename in tqdm(glob.glob(sys.argv[1]), 'converting: '):
        new_filename = filename[:-4] + '_edited.txt'

        # for each file, open and create edited
        with open(filename) as file, open(new_filename, 'w') as new_file:
            data = file.read()

            # file header
            new_file.write('#==========\n' + '# dieharder test\n' + '#==========\n'
                           + 'type: d\n'
                           + 'count: ' + str(len(data) % 32) + '\n'
                           + 'numbit: 32\n')

            while len(data) > 31:
                new_file.write(str(int(data[0:31], 2)) + '\n')
                data = data[32:]


if __name__ == '__main__':
    main()
