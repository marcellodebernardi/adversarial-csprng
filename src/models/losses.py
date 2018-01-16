"""The losses.py module defines symbolic TensorFlow functions that
can be used as custom loss functions for Keras/TensorFlow models. In
particular, it defines appropriate loss functions for the adversarially
trained generator, predictor, and discriminator networks.
"""

import utils
import tensorflow as tf


def loss_adv(predictor_loss_function):
    """Loss function for the adversarial network, used to train the
    generator."""
    def loss(true, pred):
        return tf.subtract(tf.ones(tf.shape(pred)), predictor_loss_function(true, pred))
    return loss


def loss_predictor(max_value):
    """Returns a loss function for the discriminator network.
    The maximum value parameter is used to normalize the distance
    between the predictor's output and the correct output.
    """
    def loss(true, pred):
        return tf.div(tf.abs(tf.subtract(true, pred)), max_value)
    return loss


def loss_discriminator(true, pred):
    """Loss function for the discriminative adversary"""
    return tf.abs(tf.subtract(true, pred))
