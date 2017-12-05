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

import utils
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, SimpleRNN, LSTM, Lambda
from keras.activations import linear
from models.activations import modular_activation
from models.losses import loss_pnb, loss_disc
from models.operations import drop_last_bit
from models.network import Network


SEED_LENGTH = 32
OUTPUT_LENGTH = 256
EPOCHS = 1000
USE_ORACLE = False


def main():
    """Run the full training and evaluation loop."""
    #  create nets
    gen = create_generator()
    disc = create_discriminator()
    gan = connect_models(gen, disc)

    seed = np.array([utils.get_random_seed(SEED_LENGTH)])
    dummy_output = np.array([utils.get_random_seed(OUTPUT_LENGTH)])

    # train
    gen.get_model().fit(seed, dummy_output, epochs=1)

    # save model with model.to_json, model.save, model.save_weights


def create_generator() -> Network:
    """Returns a Network object encapsulating a compiled Keras model
    that represents the generator component of the GAN."""
    return Network()\
        .with_optimizer('adagrad')\
        .with_loss_function(loss_pnb(OUTPUT_LENGTH))\
        .with_inputs(Input(shape=(SEED_LENGTH,)))\
        .add_layer(Dense(OUTPUT_LENGTH, activation=modular_activation(2)))\
        .add_layer(Dense(OUTPUT_LENGTH, activation=modular_activation(2)))\
        .add_layer(Dense(OUTPUT_LENGTH, activation=modular_activation(2)))\
        .compile()


def create_discriminator() -> Network:
    """Returns a Network object encapsulating a compiled Keras model
    that represents the discriminator component of the GAN"""
    return Network()\
        .with_optimizer('adagrad')\
        .with_loss_function(loss_disc)\
        .with_inputs(Input(shape=(OUTPUT_LENGTH - 1,)))\
        .add_layer(Dense(OUTPUT_LENGTH, activation=linear))\
        .add_layer(Dense(OUTPUT_LENGTH, activation=linear))\
        .add_layer(Dense(1, activation=linear))\
        .compile()


def connect_models(generator: Network, discriminator: Network) -> Network:
    """Connects the generator and discriminator models into a new Keras
    model by adding an intermediate layer between them that removes the
    last element from the output produces by the generator."""
    return Network()\
        .with_optimizer('adagrad')\
        .with_loss_function(loss_disc)\
        .with_inputs(generator.get_input_tensor())\
        .add_layer(generator.get_model())\
        .add_layer(Lambda(drop_last_bit(original_size=OUTPUT_LENGTH)))\
        .add_layer(discriminator.get_model())\
        .compile()


def set_trainable(model: Network, trainable: bool) -> None:
    """Sets the weights of a model to be unmodifiable. Used to freeze
    a model into a particular weight configuration."""
    model.get_model().trainable = trainable
    for layer in model.get_model().layers:
        layer.trainable = trainable


if __name__ == "__main__":
    main()
