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
and evaluation of the neural network components involved in this project.
"""

import datetime


def print_step(step, gen_loss, opp_loss):
    """ Prints debugging information for a single epoch during training.

        :param step: current step number
        :param gen_loss: generator loss at current step
        :param opp_loss: opponent loss at current step
    """
    print('Step ' + str(step) + ', ' + str(datetime.datetime.utcnow())
          + ' - Current loss: GEN %f ' % gen_loss
          + ' OPP %f ' % opp_loss
          + '-- [done]')
