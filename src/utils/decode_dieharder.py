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
For storage limitation reasons, the .
"""

import sys
import glob
from tqdm import tqdm


def main():
    for filename in tqdm(glob.glob(sys.argv[1]), 'decoding (dieharder): '):
        new_filename = filename[:-4] + '_dieharder.txt'

        # for each file, open and create edited
        with open(filename) as file, open(new_filename, 'w') as new_file:
            data = file.readlines()

            # file header
            new_file.write('#==========\n' + '# dieharder test\n' + '#==========\n'
                           + 'type: d\n'
                           + 'count: ' + str(int(len(data) / 2)) + '\n'
                           + 'numbit: 32\n')

            for number in range(len(data)):
                if number % 2 == 0:
                    new_file.write(str(int(data[number], 16)))
                else:
                    new_file.write(str(int(data[number], 16)) + '\n')


if __name__ == '__main__':
    main()
