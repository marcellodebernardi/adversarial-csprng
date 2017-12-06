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
BATCH_SIZE = 32
EPOCHS = 1000
USE_ORACLE = False


def main():
    """Run the full training and evaluation loop."""
    seed = utils.get_random_seed(SEED_LENGTH)

    #  create nets
    gen = create_generator()
    disc = create_discriminator()
    gan = connect_models(gen, disc)

    # train
    train_gan(gan, gen, disc, seed)
    # save model with model.to_json, model.save, model.save_weights


def create_generator() -> Network:
    """Returns a Network object encapsulating a compiled Keras model
    that represents the generator component of the GAN."""
    return Network()\
        .with_optimizer('adagrad')\
        .with_loss_function('binary_crossentropy')\
        .with_inputs(Input(shape=(SEED_LENGTH,)))\
        .add_layer(Dense(OUTPUT_LENGTH, activation=modular_activation(2)))\
        .add_layer(Dense(OUTPUT_LENGTH, activation=modular_activation(2)))\
        .add_layer(Dense(OUTPUT_LENGTH, activation=modular_activation(2)))\
        .compile()


def create_discriminator() -> Network:
    """Returns a Network object encapsulating a compiled Keras model
    that represents the discriminator component of the GAN"""
    # optimizer = 'rmsprop',
    # loss = 'binary_crossentropy',
    # metrics = ['accuracy']
    return Network()\
        .with_optimizer('adagrad')\
        .with_loss_function('binary_crossentropy')\
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
        .with_loss_function('binary_crossentropy')\
        .with_inputs(generator.get_input_tensor())\
        .add_layer(generator.get_model())\
        .add_layer(Lambda(drop_last_bit(original_size=OUTPUT_LENGTH, batch_size=BATCH_SIZE)))\
        .add_layer(discriminator.get_model())\
        .compile()


def train_gan(gan: Network, gen: Network, disc: Network, seed, epochs=500):
    """Performs end-to-end training of the GAN model."""
    seed_batch = utils.form_seed_batch(seed, BATCH_SIZE)
    for e in range(epochs):
        # todo train discriminator, aim is to get discriminator to discern better

        # todo train generator, aim is to compute loss on generated inputs
        print(gan.get_model().train_on_batch(seed_batch, generate_correct_nb(gen, seed, 32)))


def train_disc(disc: Network, input_data, output_data, epochs=500):
    """Used to perform pre-training on the discriminator only."""
    # todo decide on batch size
    # todo decide how to pre-train
    disc.trainable().get_model().fit(input_data, output_data, epochs)


def generate_correct_nb(gen: Network, seed, batch_size):
    """Generates a batch of final sequence bits from the generator.
    These are used as the 'correct' values that the discriminator
    should be outputting during training."""
    # todo prediction will probably affect state of RNN layers
    final_bit_array = np.empty((batch_size, 1))
    seed = np.array([seed])
    for i in range(len(final_bit_array)):
        np.append(final_bit_array, gen.get_model().predict(seed)[0][OUTPUT_LENGTH - 1])
    return final_bit_array


def generate_noise(batch_size=32, length=255):
    noise = np.empty(shape=(batch_size, length), dtype=np.int32)
    for i in range(batch_size):
        for j in range(length):
            noise[i, j] = 0


if __name__ == "__main__":
    main()
