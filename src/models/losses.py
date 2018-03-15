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
The losses.py module defines symbolic TensorFlow functions that
can be used as custom loss functions for Keras/TensorFlow models. In
particular, it defines appropriate loss functions for the adversarially
trained generator, predictor, and discriminator networks.
"""

import tensorflow as tf


def loss_disc_gan(true, pred):
    """Loss function for the adversarial network, used to train the
    generator."""
    return tf.negative(loss_discriminator(true, pred))


def loss_discriminator(true, pred):
    """ Loss function for the discriminative adversary, which computes
    the absolute difference between the true label and the predicted
    label of a sequence. """
    return tf.abs(tf.subtract(true, pred))


def loss_pred_gan(true, pred):
    return tf.negative(loss_predictor(true, pred))


def loss_predictor(true, pred):
    """Returns a loss function for the discriminator network.
    The maximum value parameter is used to normalize the distance
    between the predictor's output and the correct output.
    """
    return tf.squared_difference(true, pred)
