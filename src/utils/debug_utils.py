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
This modules provides utilities for debugging the compilation, training,
and evaluation of the neural network models involved in this project.
"""

from keras import Model
import datetime


def print_gan(generator: Model, adversary: Model, gan: Model):
    """ Prints out a summary of the network structure for a generative
    adversarial network. """
    print('Compiled network for ' + gan.name + ', summarizing:\n')
    generator.summary()
    print('\n')
    adversary.summary()
    print('\n')
    gan.summary()
    print('==================================================\n')


def print_pretrain(x_data, y_labels):
    """ Prints debugging information for the pre-training phase. """
    print('DATA:')
    print(x_data)
    print('LABELS:')
    print(y_labels)
    print('==================================================\n')


def print_epoch(epoch, inputs=None, gen_out=None, opp_out=None, gen_loss=None, opp_loss=None):
    """ Prints debugging information for a single epoch during training. """
    print('Epoch ' + str(epoch) + ' - ' + str(datetime.datetime.utcnow()) + ':')
    if inputs is not None:
        print('Inputs:')
        print(inputs)
    if gen_out is not None:
        print('Generator Output:')
        print(gen_out)
    if opp_out is not None:
        print('Adversary Output:')
        print(opp_out)
    if gen_loss is not None and opp_loss is not None:
        print('Losses: ' + str(gen_loss) + ' (gen), ' + str(opp_loss) + ' (opp)')
    print('==================================================\n')
