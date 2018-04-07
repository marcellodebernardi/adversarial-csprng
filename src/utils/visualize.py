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
This module provides utility functions for visualizing various metrics
pertaining to the trained models, including loss functions, histograms
of the model weights, etc.
"""

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from utils.operations import flatten


def plot_pretrain_loss(history, fname):
    """Plot a line chart of the adversary's loss during pre-training."""
    plt.plot(history.history['loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ticklabel_format(useOffset=False)
    plt.savefig(fname)
    plt.clf()


def plot_train_loss(generator_loss, adversary_loss, fname):
    """Plot a line chart of the generator's and adversary's losses
    during training. """
    plt.plot(generator_loss)
    plt.plot(adversary_loss)
    plt.ylabel('Loss')
    plt.ylabel('Epoch')
    plt.legend(['Generative loss', 'Adversary loss'])
    plt.ticklabel_format(useOffset=False)
    plt.savefig(fname)
    plt.clf()


def plot_output_histogram(values, fname):
    """Plot a histogram of the output values for one seed. """
    values = flatten(values)
    plt.hist(values, bins=int(abs((max(values) - min(values))*3)))
    plt.title('Generator Output Frequency Distribution')
    plt.xlabel('Output')
    plt.ylabel('Frequency')
    plt.savefig(fname)
    plt.clf()


def plot_output_sequence(values, fname):
    """Plot a line displaying the sequence of output values
    for a trained generator, for one seed, in temporal order."""
    plt.plot(flatten(values))
    plt.ylabel('Output')
    plt.xlabel('Position in Sequence')
    plt.savefig(fname)
    plt.clf()
