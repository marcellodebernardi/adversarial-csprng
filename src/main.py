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

import os
import utils
import numpy as np
from keras.layers import Input, Dense, SimpleRNN, LSTM
from keras.losses import mean_absolute_error
from keras.activations import linear
from models.activations import modular_activation
from models.generator import Generator


SEED_LENGTH = 32
OUTPUT_LENGTH = 256
USE_ORACLES = [False, False]
MAX_TRAINING_CYCLES = 10000
ITERS_PER_ACTOR = 1
DISCRIMINATOR_MULTIPLIER = 2
# logging
print_frequency = 200


def main():
    """Run the full training and evaluation loop"""
    #  create nets
    gen = create_generator()
    print(utils.get_random_seed(SEED_LENGTH))
    # train
    gen.get_model().fit(utils.get_random_seed(SEED_LENGTH), utils.get_random_seed(OUTPUT_LENGTH), epochs=1000)
    print(gen.get_model().predict(utils.get_random_seed(SEED_LENGTH)))


def create_generator():
    gen = Generator()\
        .with_optimizer('adagrad')\
        .with_loss_function(mean_absolute_error)\
        .with_inputs(Input(shape=(SEED_LENGTH,)))
    gen.add_layer(Dense(10, activation=linear))
    gen.add_layer(Dense(10, activation=linear))
    gen.add_layer(Dense(OUTPUT_LENGTH, activation=linear))
    gen.compile()
    return gen


def create_discriminator():
    pass


if __name__ == "__main__":
    main()
