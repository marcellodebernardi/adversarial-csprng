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
import traceback
import math
import numpy as np
import tensorflow as tf
from utils import utils
from utils.operation_utils import detach_all_last, set_trainable, flatten
from utils.input_utils import get_inputs, get_sequences
from utils.vis_utils import *
from utils.debug_utils import print_gan, print_pretrain, print_epoch
from keras import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, LSTM, Lambda, Reshape, Flatten
from keras.activations import linear
from keras.optimizers import adagrad
from models.activations import modulo
from models.operations import drop_last_value
from models.losses import loss_discriminator, loss_predictor, loss_disc_gan, loss_pred_gan

# main settings
HPC_TRAIN = '-t' not in sys.argv  # set to true when training on HPC to collect data

# HYPER-PARAMETERS
OUTPUT_SIZE = 8
OUTPUT_RANGE = 15
OUTPUT_BITS = 4
BATCH_SIZE = 32 if HPC_TRAIN else 4  # seeds in a single batch
BATCHES = 32 if HPC_TRAIN else 10  # batches in complete dataset
LEARNING_RATE = 0.0008
CLIP_VALUE = 0.05
DATA_TYPE = tf.float64

# losses and optimizers
DIEGO_OPT = adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE)
DIEGO_LOSS = loss_discriminator
DISC_GAN_OPT = adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE)
DISC_GAN_LOSS = loss_disc_gan
PRIYA_OPT = adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE)
PRIYA_LOSS = loss_predictor(OUTPUT_RANGE)
PRED_GAN_OPT = adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE)
PRED_GAN_LOSS = loss_pred_gan(OUTPUT_RANGE)
UNUSED_OPT = 'adagrad'
UNUSED_LOSS = 'binary_crossentropy'

# training settings
PRETRAIN = '-nopretrain' not in sys.argv
TRAIN = ['-nodisc' not in sys.argv, '-nopred' not in sys.argv]
EPOCHS = 100000 if HPC_TRAIN else 40
PRE_EPOCHS = 1000 if PRETRAIN and HPC_TRAIN else 5 if PRETRAIN else 0
ADVERSARY_MULT = 3
RECOMPILE = '-rec' in sys.argv
REFRESH_DATASET = '-ref' in sys.argv
SEND_REPORT = '-noemail' not in sys.argv

# logging and evaluation
EVAL_SEED = np.array([[1, 1], [1, 2], [1, 3]])
LOG_EVERY_N = 100 if HPC_TRAIN else 1
PLOT_DIR = '../output/plots/'


def main():
    """ Constructs the neural networks, trains them, and logs
    all relevant information."""
    if '-cpu' in sys.argv:
        with tf.device('/cpu:0'):
            # train discriminative GAN
            if TRAIN[0]:
                run_discgan()
            # train predictive GAN
            if TRAIN[1]:
                run_predgan()
    else:
        # train discriminative GAN
        if TRAIN[0]:
            run_discgan()
        # train predictive GAN
        if TRAIN[1]:
            run_predgan()

    # send off email report
    if SEND_REPORT:
        utils.email_report(BATCH_SIZE, BATCHES, EPOCHS, PRE_EPOCHS)


def run_discgan():
    """Constructs and trains the discriminative GAN consisting of
    Jerry and Diego."""
    # construct models
    jerry, diego, discgan = construct_discgan(construct_adversary_conv)
    if not HPC_TRAIN:
        print_gan(jerry, diego, discgan)

    # pre-train Diego
    diego_x, diego_y = get_sequences(jerry, get_inputs(BATCH_SIZE * BATCHES, OUTPUT_RANGE)[:-1], OUTPUT_SIZE,
                                     OUTPUT_RANGE)
    if not HPC_TRAIN:
        print_pretrain(diego_x, diego_y)
    plot_pretrain_loss(diego.fit(diego_x, diego_y, BATCH_SIZE, PRE_EPOCHS, verbose=1),
                       PLOT_DIR + 'diego_pretrain_loss.pdf')

    # train both networks in turn
    dataset = get_inputs(BATCH_SIZE * BATCHES, OUTPUT_RANGE)
    jerry_x = np.array([dataset[0], dataset[1]]).transpose()
    jerry_y = dataset[2]
    diego_x, diego_y = get_sequences(jerry, jerry_x, OUTPUT_SIZE, OUTPUT_RANGE)
    jerry_loss, diego_loss = np.zeros(EPOCHS), np.zeros(EPOCHS)

    # iterate over entire dataset
    try:
        for epoch in range(EPOCHS):
            if REFRESH_DATASET:
                diego_x, diego_y = get_sequences(jerry, jerry_x, OUTPUT_SIZE, OUTPUT_RANGE)

            # alternate train
            set_trainable(diego, DIEGO_OPT, DIEGO_LOSS, RECOMPILE)
            diego_loss[epoch] = np.mean(
                diego.fit(diego_x, diego_y, BATCH_SIZE, ADVERSARY_MULT, verbose=0).history['loss'])
            set_trainable(diego, DIEGO_OPT, DIEGO_LOSS, RECOMPILE, False)
            jerry_loss[epoch] = np.mean(discgan.fit(jerry_x, jerry_y, BATCH_SIZE, verbose=0).history['loss'])

            # log debug info to console
            if not HPC_TRAIN or epoch % LOG_EVERY_N == 0:
                print_epoch(epoch, gen_loss=jerry_loss[epoch], opp_loss=diego_loss[epoch])
            # check for NaNs
            if math.isnan(jerry_loss[epoch]) or math.isnan(diego_loss[epoch]):
                raise ValueError()

    except ValueError:
        traceback.print_exc()
    plot_train_loss(jerry_loss, diego_loss, PLOT_DIR + 'discgan_train_loss.pdf')

    # generate outputs for one seed
    values = flatten(jerry.predict(EVAL_SEED))
    plot_output_histogram(values, PLOT_DIR + 'discgan_jerry_output_distribution.pdf')
    plot_output_sequence(values, PLOT_DIR + 'discgan_jerry_output_sequence.pdf')
    plot_network_weights(flatten(jerry.get_weights()), PLOT_DIR + 'jerry_weights.pdf')
    utils.generate_output_file(values, OUTPUT_BITS, '../output/sequences/jerry.txt')


def run_predgan():
    """Constructs and trains the predictive GAN consisting of
    Janice and priya."""
    janice, priya, predgan = construct_predgan(construct_adversary_conv)
    if not HPC_TRAIN:
        print_gan(janice, priya, predgan)

    # pretrain priya
    priya_x, priya_y = detach_all_last(
        janice.predict(np.array(get_inputs(BATCH_SIZE * BATCHES, OUTPUT_RANGE)[:-1]).transpose()))
    plot_pretrain_loss(
        priya.fit(priya_x, priya_y, BATCH_SIZE, PRE_EPOCHS, verbose=1),
        PLOT_DIR + 'priya_pretrain_loss.pdf')

    # main training procedure
    janice_x = np.array(get_inputs(BATCH_SIZE * BATCHES, OUTPUT_RANGE)[:-1]).transpose()
    priya_x, priya_y = detach_all_last(janice.predict(janice_x))
    janice_loss, priya_loss = np.zeros(EPOCHS), np.zeros(EPOCHS)
    # iterate over entire dataset
    try:
        for epoch in range(EPOCHS):
            if REFRESH_DATASET:
                # janice_x = get_seed_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)[0]
                priya_x, priya_y = detach_all_last(janice.predict(janice_x))

            # train both networks on entire dataset
            set_trainable(priya, PRIYA_OPT, PRIYA_LOSS, RECOMPILE)
            priya_loss[epoch] = np.mean(
                priya.fit(priya_x, priya_y, BATCH_SIZE, ADVERSARY_MULT, verbose=0).history['loss'])
            set_trainable(priya, PRIYA_OPT, PRIYA_LOSS, RECOMPILE, False)
            janice_loss[epoch] = np.mean(predgan.fit(janice_x, priya_y, BATCH_SIZE, verbose=0).history['loss'])

            # update and log loss value
            if not HPC_TRAIN or epoch % LOG_EVERY_N == 0:
                print_epoch(epoch, gen_loss=janice_loss[epoch], opp_loss=priya_loss[epoch])
            # check for NaNs
            if math.isnan(janice_loss[epoch]) or math.isnan(priya_loss[epoch]):
                raise ValueError()

    except ValueError:
        traceback.print_exc()

    # end of training results collection
    plot_train_loss(janice_loss, priya_loss, PLOT_DIR + 'predgan_train_loss.pdf')

    # produce output for one seed
    output_values = flatten(janice.predict(EVAL_SEED))
    plot_output_histogram(output_values, PLOT_DIR + 'predgan_janice_output_distribution.pdf')
    plot_output_sequence(output_values, PLOT_DIR + 'predgan_janice_output_sequence.pdf')
    plot_network_weights(flatten(janice.get_weights()), PLOT_DIR + 'janice_weights.pdf')
    utils.generate_output_file(output_values, OUTPUT_BITS, '../output/sequences/janice.txt')


def construct_discgan(constructor):
    """Defines and compiles the models for Jerry, Diego, and the connected discgan."""
    # define Jerry
    jerry_input, jerry = construct_generator('jerry')
    # define Diego
    diego_input, diego = constructor(OUTPUT_SIZE, DIEGO_OPT, DIEGO_LOSS, 'diego')
    # define the connected GAN
    discgan_output = jerry(jerry_input)
    discgan_output = diego(discgan_output)
    discgan = Model(jerry_input, discgan_output)
    discgan.compile(DISC_GAN_OPT, DISC_GAN_LOSS)
    plot_network_graphs(discgan, 'discriminative_gan')
    return jerry, diego, discgan


def construct_predgan(constructor):
    """Defines and compiles the models for Janice, Priya, and the connected predgan. """
    # define janice
    janice_input, janice = construct_generator('janice')
    # define priya
    priya_input, priya = constructor(OUTPUT_SIZE - 1, PRIYA_OPT, PRIYA_LOSS, 'priya')
    # connect GAN
    output_predgan = janice(janice_input)
    output_predgan = Lambda(
        drop_last_value(OUTPUT_SIZE, BATCH_SIZE),
        name='adversarial_drop_last_value')(output_predgan)
    output_predgan = priya(output_predgan)
    predgan = Model(janice_input, output_predgan, name='predictive_gan')
    predgan.compile(PRED_GAN_OPT, PRED_GAN_LOSS)
    plot_network_graphs(predgan, 'predictive_gan')
    return janice, priya, predgan


def construct_generator(name: str):
    generator_input = Input(shape=(2,))
    generator_output = Dense(OUTPUT_SIZE, activation=linear)(generator_input)
    generator_output = Dense(OUTPUT_SIZE, activation=linear)(generator_output)
    generator_output = Dense(OUTPUT_SIZE, activation=modulo(OUTPUT_RANGE))(generator_output)
    generator = Model(generator_input, generator_output, name=name)

    generator.compile(UNUSED_OPT, UNUSED_LOSS)
    plot_network_graphs(generator, name)
    utils.save_configuration(generator, name)
    return generator_input, generator


def construct_adversary_conv(input_size, optimizer, loss, name: str):
    inputs = Input((input_size,))
    outputs = Reshape(target_shape=(input_size, 1))(inputs)
    outputs = Conv1D(filters=2, kernel_size=2, strides=1, padding='same', activation=linear)(outputs)
    outputs = Conv1D(filters=2, kernel_size=2, strides=1, padding='same', activation=linear)(outputs)
    outputs = MaxPooling1D(2)(outputs)
    outputs = Conv1D(filters=4, kernel_size=2, strides=1, padding='same', activation=linear)(outputs)
    outputs = Conv1D(filters=4, kernel_size=2, strides=1, padding='same', activation=linear)(outputs)
    outputs = MaxPooling1D(2)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2, activation=linear)(outputs)
    outputs = Dense(1, activation=linear)(outputs)
    discriminator = Model(inputs, outputs)

    discriminator.compile(optimizer, loss)
    plot_network_graphs(discriminator, name)
    return inputs, discriminator


def construct_adversary_lstm(input_size, optimizer, loss, name: str):
    inputs = Input((input_size,))
    outputs = Reshape(target_shape=(1, input_size))(inputs)
    outputs = LSTM(input_size, return_sequences=True)(outputs)
    outputs = LSTM(input_size, return_sequences=True)(outputs)
    outputs = LSTM(input_size, return_sequences=True)(outputs)
    outputs = Dense(input_size, activation=linear)(outputs)
    outputs = Dense(2, activation=linear)(outputs)
    outputs = Dense(1, activation=linear)(outputs)
    discriminator = Model(inputs, outputs)
    discriminator.compile(optimizer, loss)

    discriminator.compile(DIEGO_OPT, DIEGO_LOSS)
    plot_network_graphs(discriminator, name)
    return inputs, discriminator


def construct_adversary_convlstm(input_size, optimizer, loss, name: str):
    inputs = Input((input_size,))
    outputs = Reshape(target_shape=(input_size, 1))(inputs)
    outputs = Conv1D(filters=2, kernel_size=2, strides=1, padding='same', activation=linear)(outputs)
    outputs = Conv1D(filters=2, kernel_size=2, strides=1, padding='same', activation=linear)(outputs)
    outputs = MaxPooling1D(2)(outputs)
    outputs = LSTM(input_size, return_sequences=True)(outputs)
    outputs = LSTM(input_size, return_sequences=True)(outputs)
    outputs = LSTM(input_size, return_sequences=True)(outputs)
    outputs = Dense(2, activation=linear)(outputs)
    outputs = Dense(1, activation=linear)(outputs)
    discriminator = Model(inputs, outputs)
    discriminator.compile(optimizer, loss)

    discriminator.compile(DIEGO_OPT, DIEGO_LOSS)
    plot_network_graphs(discriminator, name)
    return inputs, discriminator


if __name__ == '__main__':
    main()
