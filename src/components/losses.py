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

import tensorflow as tf


def generator_classification_loss(true: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    """ Loss function for the generator in the discriminative case. Applies the function
    L(t, p) = 1 - |t - p|
    where normally t would always be 0. Thus we have L(0, 0) = 1 and L(0, 1) = 0.

    :param true: the actual class of the generator output
    :param pred: the class predicted by the discriminator
    """
    return tf.subtract(
        tf.ones(shape=tf.shape(true), dtype=true.dtype),
        tf.subtract(pred, true))


def discriminator_classification_loss(true: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    """ Loss function for the discriminator in the standard GAN. Applies the function
    L(t, p) = |t - p|
    where a completely incorrect classification would result in a loss of 1. In other
    words, we have L(1, 0) = 1, L(0, 1) = 1, L(0, 0) = 0, and L(1, 1) = 0
    """
    return tf.abs(tf.subtract(true, pred))
