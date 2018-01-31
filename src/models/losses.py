"""The losses.py module defines symbolic TensorFlow functions that
can be used as custom loss functions for Keras/TensorFlow models. In
particular, it defines appropriate loss functions for the adversarially
trained generator, predictor, and discriminator networks.
"""

import utils
import tensorflow as tf


def loss_disc_gan(true, pred):
    """Loss function for the adversarial network, used to train the
    generator."""
    # return tf.ones(tf.shape(pred))
    return tf.subtract(tf.ones(tf.shape(pred), dtype=tf.float32), tf.abs(tf.subtract(true, pred)))


def loss_discriminator(true, pred):
    """Loss function for the discriminative adversary"""
    return tf.abs(tf.subtract(true, pred))


def loss_pred_gan(max_value):
    def loss(true, pred):
        return tf.subtract(tf.ones(tf.shape(pred), dtype=tf.float32), tf.div(tf.abs(tf.subtract(true, pred)), max_value))
    return loss


def loss_predictor(max_value):
    """Returns a loss function for the discriminator network.
    The maximum value parameter is used to normalize the distance
    between the predictor's output and the correct output.
    """
    def loss(true, pred):
        return tf.div(tf.abs(tf.subtract(true, pred)), max_value)
    return loss
