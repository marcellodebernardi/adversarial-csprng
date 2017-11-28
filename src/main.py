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
from keras.layers import Input, Dense, SimpleRNN, LSTM
from keras.activations import linear
from models.generator import Generator
from models.activations import modular_activation
from models.losses import loss_pnb


SEED_LENGTH = 32
OUTPUT_LENGTH = 256
EPOCHS = 1000
USE_ORACLE = False


def main():
    """Run the full training and evaluation loop"""
    #  create nets
    gen = create_generator()
    seed = np.array([utils.get_random_seed(SEED_LENGTH)])
    dummy_output = np.array([utils.get_random_seed(OUTPUT_LENGTH)])

    # train
    gen.get_model().fit(seed, dummy_output, epochs=EPOCHS)


def create_generator():
    gen = Generator()\
        .with_optimizer('adagrad')\
        .with_loss_function(loss_pnb)\
        .with_inputs(Input(shape=(SEED_LENGTH,)))
    gen.add_layer(Dense(OUTPUT_LENGTH, activation=modular_activation(2)))
    gen.add_layer(Dense(OUTPUT_LENGTH, activation=modular_activation(2)))
    gen.add_layer(Dense(OUTPUT_LENGTH, activation=modular_activation(2)))
    gen.compile()
    return gen


def create_discriminator():
    pass


if __name__ == "__main__":
    main()
