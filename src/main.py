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
This program creates four neural networks, which form two
generative adversarial networks (GAN). The aim of both GANs
is to train the generator to output pseudo-random number
sequences that are indistinguishable from truly random ones.

The first GAN, which shall be referred to as the discriminative
GAN, consists of two networks termed Jerry (the generator) and
Diego (the discriminator). Jerry produces sequences of real
numbers which Diego attempts to distinguish from truly random
sequences. This is the standard generative adversarial
network.

The second GAN, which shall be referred to as the predictive
GAN, consists of two networks termed Janice (the generator)
and Priya (the predictor). Janice produces sequences of real
numbers in the same fashion as Jerry. Priya receives as
input the entire sequence produced by Janice, except for the
last value, which it attempts to predict.

The main function defines these networks and trains them.
"""

import sys
import numpy as np
from utils import utils, vis_utils, input_utils, operation_utils
from utils.operation_utils import get_ith_batch
from tqdm import tqdm
from keras import Model
from keras.layers import Input, Dense, SimpleRNN, Reshape, Flatten, Conv1D, LSTM, Lambda
from keras.activations import linear, softmax, relu
from keras.optimizers import adagrad
from models.activations import modulo, to_integer
from models.operations import drop_last_value
from models.losses import loss_discriminator, loss_predictor, loss_adv
from evaluators import pnb


HPC_TRAIN = False                                # set to true when training on HPC to collect data
PRETRAIN = True                                 # if true, pretrain the discriminator/predictor
BATCH_SIZE = 4096 if HPC_TRAIN else 2           # seeds in a single batch
UNIQUE_SEEDS = 128 if HPC_TRAIN else 2          # unique seeds in each batch
BATCHES = 50 if HPC_TRAIN else 1                # batches in complete dataset
EPOCHS = 300000 if HPC_TRAIN else 10            # number of epochs for training
PRETRAIN_EPOCHS = 15000 if HPC_TRAIN else 5     # number of epochs for pre-training
ADVERSARY_MULT = 2                              # multiplier for training of the adversary
VAL_BITS = 8                                    # the number of bits of each output value or seed
MAX_VAL = 255                                   # number generated are between 0-MAX_VAL
OUTPUT_LENGTH = 5000 if HPC_TRAIN else 5        # number of values generated for each seed
LEARNING_RATE = 0.008
CLIP_VALUE = 0.5


def main():
    """ Constructs the neural networks, trains them, and logs
    all relevant information."""
    if "-t" in sys.argv:
        global TESTING
        TESTING = True

    discriminative_gan()
    predictive_gan()


def discriminative_gan():
    """Constructs and trains the discriminative GAN consisting of
    Jerry and Diego."""
    # define Jerry
    jerry_input = Input(shape=(1,))
    jerry_output = Dense(OUTPUT_LENGTH, activation=linear)(jerry_input)
    jerry_output = Reshape(target_shape=(5, int(OUTPUT_LENGTH / 5)))(jerry_output)
    jerry_output = SimpleRNN(int(OUTPUT_LENGTH / 5), return_sequences=True, activation=linear)(jerry_output)
    jerry_output = Flatten()(jerry_output)
    jerry_output = Dense(OUTPUT_LENGTH, activation=modulo(MAX_VAL))(jerry_output)
    jerry = Model(jerry_input, jerry_output, name='jerry')
    jerry.compile('adagrad', 'binary_crossentropy')
    vis_utils.plot_network_graphs(jerry, 'jerry')
    # define Diego
    diego_input = Input(shape=(OUTPUT_LENGTH,))
    diego_output = Dense(OUTPUT_LENGTH, activation=linear)(diego_input)
    diego_output = Reshape(target_shape=(5, int(OUTPUT_LENGTH / 5)))(diego_output)
    diego_output = Conv1D(int(OUTPUT_LENGTH / 4), 4)(diego_output)
    diego_output = Flatten()(diego_output)
    diego_output = Dense(int(OUTPUT_LENGTH / 4), activation=linear)(diego_output)
    diego_output = Dense(1, activation=softmax)(diego_output)
    diego = Model(diego_input, diego_output)
    diego.compile(adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE), loss_discriminator)
    vis_utils.plot_network_graphs(diego, 'diego')
    # define the connected GAN
    discgan_output = jerry(jerry_input)
    discgan_output = diego(discgan_output)
    discgan = Model(jerry_input, discgan_output)
    discgan.compile(adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE), loss_adv(loss_discriminator))
    vis_utils.plot_network_graphs(discgan, 'discriminative_gan')

    # pre-train Diego
    x, y = input_utils.get_discriminator_training_dataset(jerry, BATCH_SIZE, BATCHES, OUTPUT_LENGTH, MAX_VAL)
    diego.fit(x, y, batch_size=BATCH_SIZE, epochs=PRETRAIN_EPOCHS, verbose=0)

    # train both networks in turn
    for epoch in tqdm(range(EPOCHS), desc='Train jerry and diego: '):
        x_d, y_d = input_utils.get_discriminator_training_dataset(jerry, BATCH_SIZE, BATCHES, OUTPUT_LENGTH, MAX_VAL)
        x_j, y_j = input_utils.get_jerry_training_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)
        for batch in range(BATCHES):
            # train diego
            operation_utils.set_trainable(diego)
            for iteration in range(ADVERSARY_MULT):
                diego.train_on_batch(get_ith_batch(x_d, batch, BATCH_SIZE), get_ith_batch(y_d, batch, BATCH_SIZE))
            # train jerry
            operation_utils.set_trainable(diego, False)
            discgan.train_on_batch(get_ith_batch(x_j, batch, BATCH_SIZE), get_ith_batch(y_j, batch, BATCH_SIZE))
    # generate output file for one seed
    utils.generate_output_file(jerry, MAX_VAL, VAL_BITS)
    # pnb.evaluate('../sequences/jerry.txt')


def predictive_gan():
    """Constructs and trains the predictive GAN consisting of
    Janice and priya."""
    # define janice
    janice_input = Input(shape=(1,))
    janice_output = Dense(OUTPUT_LENGTH, activation=linear)(janice_input)
    janice_output = Reshape(target_shape=(5, int(OUTPUT_LENGTH / 5)))(janice_output)
    janice_output = SimpleRNN(60, return_sequences=True, activation=linear)(janice_output)
    janice_output = Flatten()(janice_output)
    janice_output = Dense(OUTPUT_LENGTH, activation=modulo(MAX_VAL))(janice_output)
    janice = Model(janice_input, janice_output, name='janice')
    janice.compile('adagrad', 'binary_crossentropy')
    vis_utils.plot_network_graphs(janice, 'janice')
    # define priya
    priya_input = Input(shape=(OUTPUT_LENGTH - 1,))
    priya_output = Dense(OUTPUT_LENGTH)(priya_input)
    priya_output = Reshape(target_shape=(5, int(OUTPUT_LENGTH / 5)))(priya_output)
    priya_output = LSTM(int(OUTPUT_LENGTH / 5), return_sequences=True, activation=linear)(priya_output)
    priya_output = Flatten()(priya_output)
    priya_output = Dense(1, activation=relu)(priya_output)
    priya = Model(priya_input, priya_output)
    priya.compile(adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE), loss_predictor(MAX_VAL))
    vis_utils.plot_network_graphs(priya, 'priya')
    # connect GAN
    output_predgan = janice(janice_input)
    output_predgan = Lambda(
        drop_last_value(OUTPUT_LENGTH, BATCH_SIZE),
        name='adversarial_drop_last_value')(output_predgan)
    output_predgan = priya(output_predgan)
    predgan = Model(janice_input, output_predgan, name='predictive_gan')
    predgan.compile(adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE), loss_adv(loss_predictor(MAX_VAL)))
    vis_utils.plot_network_graphs(predgan, 'predictive_gan')

    # pretrain priya
    seed_dataset = input_utils.get_jerry_training_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)[0]
    janice_output = janice.predict(seed_dataset)
    priya_input, priya_output = operation_utils.split_generator_outputs_batch(janice_output, 1)
    priya.fit(priya_input, priya_output, batch_size=BATCH_SIZE, epochs=PRETRAIN_EPOCHS, verbose=0)

    # train both janice and priya
    for epoch in tqdm(range(EPOCHS), desc='Train janice and priya: '):
        seed_dataset = input_utils.get_jerry_training_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)[0]
        for batch in range(BATCHES):
            janice_output = janice.predict_on_batch(get_ith_batch(seed_dataset, batch, BATCH_SIZE))
            # priya_input, priya_output = operation_utils.split_generator_output(janice_output, 1)
            priya_input, priya_output = operation_utils.split_generator_outputs_batch(janice_output, 1)
            operation_utils.set_trainable(priya)
            for i in range(ADVERSARY_MULT):
                priya.train_on_batch(priya_input, priya_output)
            # train generator
            operation_utils.set_trainable(priya, False)
            predgan.train_on_batch(get_ith_batch(seed_dataset, batch, BATCH_SIZE), priya_output)

    utils.generate_output_file(janice, MAX_VAL, VAL_BITS)
    # pnb.evaluate('../sequences/' + str(janice.name) + '.txt')


if __name__ == '__main__':
    main()
