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
import datetime
import math
import numpy as np
import tensorflow as tf
from utils import utils, vis_utils, input_utils, operation_utils
from utils.operation_utils import get_ith_batch, split_generator_outputs, set_trainable
from utils.input_utils import get_seed_dataset, get_sequences_dataset
from utils.vis_utils import *
from keras import Model
from keras.layers import Input, Dense, SimpleRNN, Reshape, Flatten, Conv1D, LSTM, Lambda
from keras.activations import linear, relu, softmax
from keras.optimizers import adagrad, sgd
from keras.losses import mean_absolute_error, binary_crossentropy, mean_absolute_percentage_error
from models.activations import modulo, diagonal_max
from models.operations import drop_last_value
from models.losses import loss_discriminator, loss_predictor, loss_disc_gan, loss_pred_gan

HPC_TRAIN = '-t' not in sys.argv  # set to true when training on HPC to collect data
TRAIN = ['-nodisc' not in sys.argv, '-nopred' not in sys.argv]  # Indicates whether discgan / predgan are to be trained
PRETRAIN = True  # if true, pretrain the discriminator/predictor
RECOMPILE = '-rec' in sys.argv  # if true, models are recompiled when set_trainable
REFRESH_DATASET = '-ref' in sys.argv  # refresh dataset at each epoch
SEND_REPORT = '-noemail' not in sys.argv  # emails results to given addresses
BATCH_SIZE = 4096 if HPC_TRAIN else 10  # seeds in a single batch
UNIQUE_SEEDS = 64 if HPC_TRAIN else 5  # unique seeds in each batch
BATCHES = 200 if HPC_TRAIN else 10  # batches in complete dataset
EPOCHS = 100000 if HPC_TRAIN else 100  # number of epochs for training
PRETRAIN_EPOCHS = 1 if '-nopretrain' in sys.argv else 20000 if HPC_TRAIN else 5  # number of epochs for pre-training
ADVERSARY_MULT = 20  # multiplier for training of the adversary
VAL_BITS = 16 if HPC_TRAIN else 4  # the number of bits of each output value or seed
MAX_VAL = 65535 if HPC_TRAIN else 15  # number generated are between 0-MAX_VAL
OUTPUT_LENGTH = 50000 if HPC_TRAIN else 5  # number of values generated for each seed
LEARNING_RATE = 0.0008
CLIP_VALUE = 0.5
# losses and optimizers
DIEGO_OPT = adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE)
DIEGO_LOSS = loss_discriminator
DISC_GAN_OPT = adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE)
DISC_GAN_LOSS = loss_disc_gan
PRIYA_OPT = adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE)
PRIYA_LOSS = loss_predictor(MAX_VAL)
PRED_GAN_OPT = adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE)
PRED_GAN_LOSS = loss_pred_gan(MAX_VAL)
UNUSED_OPT = 'adagrad'
UNUSED_LOSS = 'binary_crossentropy'
# evaluation seed
EVAL_SEED = input_utils.get_random_sequence(1, MAX_VAL)
# logging
LOG_EVERY_N = 100 if HPC_TRAIN else 10


def main():
    """ Constructs the neural networks, trains them, and logs
    all relevant information."""
    if '-cpu' in sys.argv:
        with tf.device('/cpu:0'):
            # train discriminative GAN
            if TRAIN[0]:
                discriminative_gan()
            # train predictive GAN
            if TRAIN[1]:
                predictive_gan()
    else:
        # train discriminative GAN
        if TRAIN[0]:
            discriminative_gan()
        # train predictive GAN
        if TRAIN[1]:
            predictive_gan()

    # send off email report
    if SEND_REPORT:
        utils.email_report(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, EPOCHS, PRETRAIN_EPOCHS)


def discriminative_gan():
    """Constructs and trains the discriminative GAN consisting of
    Jerry and Diego."""
    # construct models
    jerry, diego, discgan = construct_discgan()

    # pre-train Diego
    diego_x_data, diego_y_labels = get_sequences_dataset(jerry, BATCH_SIZE, BATCHES, OUTPUT_LENGTH, MAX_VAL)
    plot_pretrain_history_loss(
        diego.fit(diego_x_data, diego_y_labels, batch_size=BATCH_SIZE, epochs=PRETRAIN_EPOCHS, verbose=0),
        '../output/plots/diego_pretrain_loss.pdf')

    # train both networks in turn
    jerry_loss, diego_loss = np.zeros(EPOCHS), np.zeros(EPOCHS)
    diego_x_data, diego_y_labels = get_sequences_dataset(jerry, BATCH_SIZE, BATCHES, OUTPUT_LENGTH, MAX_VAL)
    discgan_x_data, discgan_y_labels = get_seed_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)
    # iterate over entire dataset
    try:
        for epoch in range(EPOCHS):
            if REFRESH_DATASET:
                diego_x_data, diego_y_labels = get_sequences_dataset(jerry, BATCH_SIZE, BATCHES, OUTPUT_LENGTH, MAX_VAL)
                discgan_x_data, discgan_y_labels = get_seed_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)
            set_trainable(diego, DIEGO_OPT, DIEGO_LOSS, RECOMPILE)
            diego_loss[epoch] += np.mean(diego.fit(diego_x_data, diego_y_labels, batch_size=BATCH_SIZE,
                                                   epochs=ADVERSARY_MULT, verbose=0).history['loss'])
            set_trainable(diego, DIEGO_OPT, DIEGO_LOSS, RECOMPILE, False)
            jerry_loss[epoch] += np.mean(discgan.fit(discgan_x_data, discgan_y_labels, batch_size=BATCH_SIZE,
                                                     verbose=0).history['loss'])

            if math.isnan(jerry_loss[epoch]) or math.isnan(diego_loss[epoch]):
                raise ValueError()
            # log losses to console
            if epoch % LOG_EVERY_N == 0:
                print(
                    str(datetime.datetime.utcnow()) + 'Jerry loss: ' + str(jerry_loss[epoch]) + ' / Diego loss: ' + str(
                        diego_loss[epoch]))
    except ValueError:
        print(str(datetime.datetime.utcnow()) + " loss is nan, aborting")
    plot_train_loss(jerry_loss, diego_loss, '../output/plots/discgan_train_loss.pdf')

    # generate outputs for one seed
    values = flatten_irregular_nested_iterable(jerry.predict(EVAL_SEED))
    plot_output_histogram(values, '../output/plots/discgan_jerry_output_distribution.pdf')
    plot_output_sequence(values, '../output/plots/discgan_jerry_output_sequence.pdf')
    plot_network_weights(
        operation_utils.flatten_irregular_nested_iterable(jerry.get_weights()),
        '../output/plots/jerry_weights.pdf'
    )
    utils.generate_output_file(values, VAL_BITS, '../output/sequences/jerry.txt')


def predictive_gan():
    """Constructs and trains the predictive GAN consisting of
    Janice and priya."""
    janice, priya, predgan = construct_predgan()

    # pretrain priya
    priya_x_data, priya_y_labels = split_generator_outputs(
        janice.predict(get_seed_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)[0]))
    plot_pretrain_history_loss(
        priya.fit(priya_x_data, priya_y_labels, BATCH_SIZE, PRETRAIN_EPOCHS, verbose=0),
        '../output/plots/priya_pretrain_loss.pdf')

    # main training procedure
    janice_loss, priya_loss = np.zeros(EPOCHS), np.zeros(EPOCHS)
    seed_dataset = get_seed_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)[0]
    priya_x_data, priya_y_labels = split_generator_outputs(janice.predict(seed_dataset))
    # iterate over entire dataset
    try:
        for epoch in range(EPOCHS):
            if REFRESH_DATASET:
                seed_dataset = get_seed_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)[0]
                priya_x_data, priya_y_labels = split_generator_outputs(janice.predict(seed_dataset))
            # iterate over portions of dataset
            for batch in range(BATCHES):
                # train predictor
                set_trainable(priya, PRIYA_OPT, PRIYA_LOSS, RECOMPILE)
                for i in range(ADVERSARY_MULT):
                    priya_loss[epoch] += priya.fit(
                        get_ith_batch(priya_x_data, batch, BATCH_SIZE),
                        get_ith_batch(priya_y_labels, batch, BATCH_SIZE), verbose=0).history['loss']
                # train generator
                set_trainable(priya, PRIYA_OPT, PRIYA_LOSS, RECOMPILE, False)
                batch_train_pred = predgan.fit(
                    get_ith_batch(seed_dataset, batch, BATCH_SIZE),
                    get_ith_batch(priya_y_labels, batch, BATCH_SIZE), verbose=0).history['loss']
                janice_loss[epoch] = batch_train_pred[0]
                if math.isnan(janice_loss[epoch]) or math.isnan(priya_loss[epoch]):
                    raise ValueError()
            # update and log loss value
            janice_loss[epoch] /= BATCHES
            priya_loss[epoch] /= (BATCHES * ADVERSARY_MULT)
            if epoch % LOG_EVERY_N == 0:
                print(str(datetime.datetime.utcnow()) + 'Janice loss: ' + str(
                    janice_loss[epoch]) + ' / Priya loss: ' + str(priya_loss[epoch]))
    except ValueError:
        print(str(datetime.datetime.utcnow()) + " loss is nan, aborting")

    # end of training results collection
    plot_train_loss(janice_loss, priya_loss, '../output/plots/predgan_train_loss.pdf')

    # produce output for one seed
    output_values = flatten_irregular_nested_iterable(janice.predict(EVAL_SEED))
    plot_output_histogram(output_values, '../output/plots/predgan_janice_output_distribution.pdf')
    plot_output_sequence(output_values, '../output/plots/predgan_janice_output_sequence.pdf')
    plot_network_weights(
        flatten_irregular_nested_iterable(janice.get_weights()),
        '../output/plots/janice_weights.pdf'
    )
    utils.generate_output_file(output_values, VAL_BITS, '../output/sequences/janice.txt')


def construct_discgan():
    """Defines and compiles the models for Jerry, Diego, and the connected discgan."""
    # define Jerry
    jerry_input, jerry = construct_generator('jerry')
    # define Diego
    diego_input = Input(shape=(OUTPUT_LENGTH,))
    diego_output = Dense(OUTPUT_LENGTH)(diego_input)
    diego_output = Reshape(target_shape=(5, int(OUTPUT_LENGTH / 5)))(diego_output)
    diego_output = Conv1D(int(OUTPUT_LENGTH / 4), 4)(diego_output)
    diego_output = Flatten()(diego_output)
    diego_output = Dense(1, activation=diagonal_max(100))(diego_output)
    diego = Model(diego_input, diego_output)
    diego.compile(DIEGO_OPT, DIEGO_LOSS)
    plot_network_graphs(diego, 'diego')
    utils.save_configuration(diego, 'diego')  # todo this occupies too much storage
    # define the connected GAN
    discgan_output = jerry(jerry_input)
    discgan_output = diego(discgan_output)
    discgan = Model(jerry_input, discgan_output)
    discgan.compile(DISC_GAN_OPT, DISC_GAN_LOSS)
    plot_network_graphs(discgan, 'discriminative_gan')
    utils.save_configuration(discgan, 'discgan')  # todo this occupies too much storage

    return jerry, diego, discgan


def construct_predgan():
    """Defines and compiles the models for Janice, Priya, and the connected predgan. """
    # define janice
    janice_input, janice = construct_generator('janice')
    # define priya
    priya_input = Input(shape=(OUTPUT_LENGTH - 1,))
    priya_output = Dense(OUTPUT_LENGTH)(priya_input)
    priya_output = Reshape(target_shape=(5, int(OUTPUT_LENGTH / 5)))(priya_output)
    priya_output = LSTM(int(OUTPUT_LENGTH / 5), return_sequences=True, activation=linear)(priya_output)
    priya_output = Flatten()(priya_output)
    priya_output = Dense(1, activation=linear)(priya_output)
    priya = Model(priya_input, priya_output)
    priya.compile(PRIYA_OPT, PRIYA_LOSS)
    plot_network_graphs(priya, 'priya')
    utils.save_configuration(priya, 'priya')
    # connect GAN
    output_predgan = janice(janice_input)
    output_predgan = Lambda(
        drop_last_value(OUTPUT_LENGTH, BATCH_SIZE),
        name='adversarial_drop_last_value')(output_predgan)
    output_predgan = priya(output_predgan)
    predgan = Model(janice_input, output_predgan, name='predictive_gan')
    predgan.compile(PRED_GAN_OPT, PRED_GAN_LOSS)
    plot_network_graphs(predgan, 'predictive_gan')
    # utils.save_configuration(predgan, 'predgan')
    return janice, priya, predgan


def construct_generator(name: str):
    generator_input = Input(shape=(1,))
    generator_output = Dense(OUTPUT_LENGTH, activation=linear)(generator_input)
    generator_output = Reshape(target_shape=(5, int(OUTPUT_LENGTH / 5)))(generator_output)
    generator_output = SimpleRNN(int(OUTPUT_LENGTH / 5), return_sequences=True, activation=linear)(generator_output)
    generator_output = Flatten()(generator_output)
    generator_output = Dense(OUTPUT_LENGTH, activation=modulo(MAX_VAL))(generator_output)
    generator = Model(generator_input, generator_output, name=name)
    generator.compile(UNUSED_OPT, UNUSED_LOSS)
    plot_network_graphs(generator, name)
    utils.save_configuration(generator, name)
    return generator_input, generator


if __name__ == '__main__':
    main()
