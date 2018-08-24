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
This module provides utility functions for visualizing various metrics
pertaining to the trained components, including loss functions, histograms
of the model weights, etc.
"""

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from utils.operations import flatten


def plot_train_loss(generator_loss, adversary_loss, filename):
    """ Plot a line chart of the generator's and adversary's losses
        during training.

        :param generator_loss: list or numpy array holding loss values for generator
        :param adversary_loss: list or numpy array holding loss values for adversary
        :param filename: name of file to draw loss plot into
    """
    plt.plot(generator_loss)
    plt.plot(adversary_loss)
    plt.ylabel('Loss')
    plt.ylabel('Epoch')
    plt.legend(['Generative loss', 'Adversary loss'])
    plt.ticklabel_format(useOffset=False)
    plt.savefig(filename)
    plt.clf()


def plot_output_histogram(values, filename):
    """ Plot a frequency histogram of the output values for one seed.

        :param values: list or numpy array containing generator outputs
        :param filename: name of file to draw histogram into
    """
    values = flatten(values)
    plt.hist(values, bins=int(abs((max(values) - min(values))*3)))
    plt.title('Generator Output Frequency Distribution')
    plt.xlabel('Output')
    plt.ylabel('Frequency')
    plt.savefig(filename)
    plt.clf()


def plot_output_sequence(values, filename):
    """ Plot a line displaying the sequence of output values
        for a trained generator, for one seed, in temporal order.

        :param values: list or numpy array containing generator outputs
        :param filename: name of file to draw line plot into    
    """
    plt.plot(flatten(values))
    plt.ylabel('Output')
    plt.xlabel('Position in Sequence')
    plt.savefig(filename)
    plt.clf()
