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

Available command line arguments:
-t              TEST MODE: runs model with reduced size and few iterations
-nopretrain     SKIP PRETRAINING: does not perform pretraining of the adversary
-nodisc         SKIP DISCRIMINATIVE GAN: does not train discriminative GAN
-nopred         SKIP PREDICTIVE GAN: does not train predictive GAN
-rec            RECOMPILE: recompiles models when changing trainability of weights
-big            BIG GENERATOR: increases width of generator hidden layers
-bound          BOUNDING CLIP ACTIVATION: uses the "bounding clip" activation function for adversary
-leakybound     LEAKY BOUNDING CLIP ACTIVATION: uses the "leaky bounding clip" activation for adversary
-lstm           LSTM: uses the lstm-only architecture for the adversary
-convlstm       CONVOLUTIONAL LSTM: uses the convolution + lstm mixed architecture for adversary
"""

import sys
import traceback
import math
import numpy as np
import tensorflow as tf
from utils.utils import log_to_file, email_report, generate_output_file, save_configuration
from utils.operation_utils import detach_all_last, set_trainable, flatten, extract_batch
from utils.input_utils import get_inputs, get_sequences, get_eval_input
from utils.vis_utils import *
from utils.debug_utils import print_gan, print_epoch
from keras import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, LSTM, Lambda, Reshape, Flatten
from keras.activations import relu
from keras.callbacks import EarlyStopping
from keras.optimizers import adam
from models.activations import bounding_clip, leaky_bounding_clip
from models.operations import drop_last_value
from models.losses import loss_discriminator, loss_predictor, loss_disc_gan, loss_pred_gan

# main settings
HPC_TRAIN = '-t' not in sys.argv  # set to true when training on HPC to collect data
ARCHITECTURE = 'lstm' if '-lstm' in sys.argv else 'convlstm' if '-convlstm' in sys.argv else 'conv'
BIG_GENERATOR = '-big' in sys.argv

# HYPER-PARAMETERS
OUTPUT_SIZE = 8
MAX_VAL = 15
OUTPUT_BITS = 4
BATCH_SIZE = 32 if HPC_TRAIN else 4  # seeds in a single batch
BATCHES = 64 if HPC_TRAIN else 10  # batches in complete dataset
LEARNING_RATE = 0.0008
CLIP_VALUE = 0.05
ALPHA = 0.01
GEN_WIDTH = 100 if BIG_GENERATOR else 10
ACTIVATION = leaky_bounding_clip(MAX_VAL, ALPHA) if '-leakybound' in sys.argv else bounding_clip(
    MAX_VAL) if '-bound' in sys.argv else relu
DATA_TYPE = tf.float64

# losses and optimizers
DIEGO_OPT = adam(lr=LEARNING_RATE, clipvalue=CLIP_VALUE)
DIEGO_LOSS = loss_discriminator
DISC_GAN_OPT = adam(lr=LEARNING_RATE, clipvalue=CLIP_VALUE)
DISC_GAN_LOSS = loss_disc_gan
PRIYA_OPT = adam(lr=LEARNING_RATE, clipvalue=CLIP_VALUE)
PRIYA_LOSS = loss_predictor
PRED_GAN_OPT = adam(lr=LEARNING_RATE, clipvalue=CLIP_VALUE)
PRED_GAN_LOSS = loss_pred_gan
UNUSED_OPT = 'adagrad'
UNUSED_LOSS = 'binary_crossentropy'

# training settings
PRETRAIN = '-nopretrain' not in sys.argv
TRAIN = ['-nodisc' not in sys.argv, '-nopred' not in sys.argv]
EPOCHS = 100000 if HPC_TRAIN else 40
PRE_EPOCHS = 1000 if PRETRAIN and HPC_TRAIN else 5 if PRETRAIN else 0
ADV_MULT = 5
RECOMPILE = '-rec' in sys.argv
SEND_REPORT = '-email' in sys.argv

# logging and evaluation
EVAL_DATA = get_eval_input(10, 50000 if HPC_TRAIN else 10)
LOG_EVERY_N = 1000 if HPC_TRAIN else 1
PLOT_DIR = '../output/plots/'
DATA_DIR = '../output/data/'
SEQN_DIR = '../output/sequences/'
MODEL_DIR = '../output/saved_models/'
GRAPH_DIR = '../output/model_graphs/'


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
        email_report(BATCH_SIZE, BATCHES, EPOCHS, PRE_EPOCHS)

    # print settings for convenience
    print('TRAINING COMPLETE')
    print('With ' + ARCHITECTURE + ' adversaries, big generator: ' + str(BIG_GENERATOR) + ', and '
          + ACTIVATION + 'activation.')


def run_discgan():
    """Constructs and trains the discriminative GAN consisting of
    Jerry and Diego."""
    # construct models
    jerry, diego, discgan = construct_discgan(select_constructor(ARCHITECTURE))
    print_gan(jerry, diego, discgan)

    # pre-train Diego
    diego_x, diego_y = get_sequences(jerry, np.array(get_inputs(BATCH_SIZE * BATCHES, MAX_VAL)[:-1]).transpose(),
                                     OUTPUT_SIZE, MAX_VAL)
    stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=0, mode='auto')
    plot_pretrain_loss(diego.fit(diego_x, diego_y, BATCH_SIZE, PRE_EPOCHS, verbose=1, callbacks=[stopping]),
                       PLOT_DIR + 'diego_pretrain_loss.pdf')

    # main training procedure
    jerry_loss, diego_loss = np.zeros(EPOCHS), np.zeros(EPOCHS)
    try:
        for epoch in range(EPOCHS):
            # training data for this epoch
            seeds, offsets, jerry_labels = get_inputs(BATCH_SIZE * BATCHES, MAX_VAL)
            jerry_inputs = np.array([seeds, offsets]).transpose()

            for batch in range(BATCHES):
                # generate batch data
                jerry_x = extract_batch(jerry_inputs, batch, int(BATCH_SIZE / 2))
                jerry_y = extract_batch(jerry_labels, batch, int(BATCH_SIZE / 2))
                diego_x, diego_y = get_sequences(jerry, jerry_x, OUTPUT_SIZE, MAX_VAL)

                # alternate train
                set_trainable(diego, DIEGO_OPT, DIEGO_LOSS, RECOMPILE)
                for i in range(ADV_MULT):
                    diego_loss[epoch] += diego.train_on_batch(diego_x, diego_y)
                set_trainable(diego, DIEGO_OPT, DIEGO_LOSS, RECOMPILE, False)
                jerry_loss[epoch] += discgan.train_on_batch(jerry_x, jerry_y)

            # log debug info to console
            diego_loss[epoch] /= (BATCHES * ADV_MULT)
            jerry_loss[epoch] /= BATCHES
            if not HPC_TRAIN or epoch % LOG_EVERY_N == 0:
                print_epoch(epoch, gen_loss=jerry_loss[epoch], opp_loss=diego_loss[epoch])
            # check for NaNs
            if math.isnan(jerry_loss[epoch]) or math.isnan(diego_loss[epoch]):
                raise ValueError()

    except ValueError:
        traceback.print_exc()

    # log training stats
    plot_train_loss(jerry_loss, diego_loss, PLOT_DIR + 'discgan_train_loss.pdf')
    log_to_file(jerry_loss, DATA_DIR + 'jerry_loss.txt')
    log_to_file(diego_loss, DATA_DIR + 'diego_loss.txt')
    log_to_file(jerry.get_weights(), DATA_DIR + 'jerry_weights.txt')
    log_to_file(diego.get_weights(), DATA_DIR + 'diego_weights.txt')
    plot_network_weights(flatten(jerry.get_weights()), PLOT_DIR + 'jerry_weights.pdf')

    # generate outputs for one seed
    values = flatten(jerry.predict(EVAL_DATA))
    log_to_file(values, DATA_DIR + 'jerry_eval_sequence.txt')
    generate_output_file(values, OUTPUT_BITS, SEQN_DIR + 'jerry.txt')
    plot_output_histogram(values, PLOT_DIR + 'discgan_jerry_output_distribution.pdf')
    plot_output_sequence(values, PLOT_DIR + 'discgan_jerry_output_sequence.pdf')


def run_predgan():
    """Constructs and trains the predictive GAN consisting of
    Janice and priya."""
    janice, priya, predgan = construct_predgan(select_constructor(ARCHITECTURE))
    if not HPC_TRAIN:
        print_gan(janice, priya, predgan)

    # pretrain priya
    priya_x, priya_y = detach_all_last(
        janice.predict(np.array(get_inputs(BATCH_SIZE * BATCHES, MAX_VAL)[:-1]).transpose()))
    stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=0, mode='auto')
    plot_pretrain_loss(
        priya.fit(priya_x, priya_y, BATCH_SIZE, PRE_EPOCHS, verbose=1, callbacks=[stopping]),
        PLOT_DIR + 'priya_pretrain_loss.pdf')

    # main training procedure
    janice_loss, priya_loss = np.zeros(EPOCHS), np.zeros(EPOCHS)
    try:
        for epoch in range(EPOCHS):
            # get data for this epoch
            janice_inputs = np.array(get_inputs(BATCH_SIZE * BATCHES, MAX_VAL)[:-1]).transpose()

            for batch in range(BATCHES):
                # generate predictions for this batch
                janice_x = extract_batch(janice_inputs, batch, BATCH_SIZE)
                priya_x, priya_y = detach_all_last(janice.predict_on_batch(janice_x))

                # train both networks on entire dataset
                set_trainable(priya, PRIYA_OPT, PRIYA_LOSS, RECOMPILE)
                for i in range(ADV_MULT):
                    priya_loss[epoch] += priya.train_on_batch(priya_x, priya_y)
                set_trainable(priya, PRIYA_OPT, PRIYA_LOSS, RECOMPILE, False)
                janice_loss[epoch] += predgan.train_on_batch(janice_x, priya_y)

            # update and log loss value
            priya_loss[epoch] /= (BATCHES * ADV_MULT)
            janice_loss[epoch] /= BATCHES
            if epoch % LOG_EVERY_N == 0:
                print_epoch(epoch, gen_loss=janice_loss[epoch], opp_loss=priya_loss[epoch])
            # check for NaNs
            if math.isnan(janice_loss[epoch]) or math.isnan(priya_loss[epoch]):
                raise ValueError()

    except ValueError:
        traceback.print_exc()

    # log training stats
    plot_train_loss(janice_loss, priya_loss, PLOT_DIR + 'predgan_train_loss.pdf')
    log_to_file(janice_loss, DATA_DIR + 'janice_loss.txt')
    log_to_file(priya_loss, DATA_DIR + 'priya_loss.txt')
    log_to_file(janice.get_weights(), DATA_DIR + 'janice_weights.txt')
    log_to_file(priya.get_weights(), DATA_DIR + 'priya_weights.txt')
    plot_network_weights(flatten(janice.get_weights()), PLOT_DIR + 'janice_weights.pdf')

    # generate outputs for one seed
    values = flatten(janice.predict(EVAL_DATA))
    log_to_file(values, DATA_DIR + 'janice_eval_sequence.txt')
    generate_output_file(values, OUTPUT_BITS, SEQN_DIR + 'janice.txt')
    plot_output_histogram(values, PLOT_DIR + 'predgan_janice_output_distribution.pdf')
    plot_output_sequence(values, PLOT_DIR + 'predgan_janice_output_sequence.pdf')


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


def select_constructor(name: str):
    return {
        'conv': construct_adversary_conv,
        'lstm': construct_adversary_lstm,
        'convlstm': construct_adversary_convlstm
    }[name]


def construct_generator(name: str):
    generator_input = Input(shape=(2,))
    generator_output = Dense(GEN_WIDTH, activation=leaky_bounding_clip(MAX_VAL, ALPHA))(generator_input)
    generator_output = Dense(GEN_WIDTH, activation=leaky_bounding_clip(MAX_VAL, ALPHA))(generator_output)
    generator_output = Dense(GEN_WIDTH, activation=leaky_bounding_clip(MAX_VAL, ALPHA))(generator_output)
    generator_output = Dense(GEN_WIDTH, activation=leaky_bounding_clip(MAX_VAL, ALPHA))(generator_output)
    generator_output = Dense(OUTPUT_SIZE, activation=leaky_bounding_clip(MAX_VAL, ALPHA))(generator_output)
    generator = Model(generator_input, generator_output, name=name)

    generator.compile(UNUSED_OPT, UNUSED_LOSS)
    plot_network_graphs(generator, name)
    save_configuration(generator, name)
    return generator_input, generator


def construct_adversary_conv(input_size, optimizer, loss, name: str):
    inputs = Input((input_size,))
    outputs = Reshape(target_shape=(input_size, 1))(inputs)
    outputs = Conv1D(filters=2, kernel_size=2, strides=1, padding='same', activation=ACTIVATION)(outputs)
    outputs = Conv1D(filters=2, kernel_size=2, strides=1, padding='same', activation=ACTIVATION)(outputs)
    outputs = MaxPooling1D(2)(outputs)
    outputs = Conv1D(filters=4, kernel_size=2, strides=1, padding='same', activation=ACTIVATION)(outputs)
    outputs = Conv1D(filters=4, kernel_size=2, strides=1, padding='same', activation=ACTIVATION)(outputs)
    outputs = MaxPooling1D(2)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2, activation=ACTIVATION)(outputs)
    outputs = Dense(1, activation=ACTIVATION)(outputs)
    discriminator = Model(inputs, outputs)

    discriminator.compile(optimizer, loss)
    plot_network_graphs(discriminator, name)
    return inputs, discriminator


def construct_adversary_lstm(input_size, optimizer, loss, name: str):
    inputs = Input((input_size,))
    outputs = Reshape(target_shape=(input_size, 1))(inputs)
    outputs = LSTM(1, return_sequences=True, activation=ACTIVATION)(outputs)
    outputs = LSTM(1, return_sequences=True, activation=ACTIVATION)(outputs)
    outputs = LSTM(1, return_sequences=True, activation=ACTIVATION)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(int(input_size / 2), activation=ACTIVATION)(outputs)
    outputs = Dense(2, activation=ACTIVATION)(outputs)
    outputs = Dense(1, activation=ACTIVATION)(outputs)
    discriminator = Model(inputs, outputs)
    discriminator.compile(optimizer, loss)

    discriminator.compile(DIEGO_OPT, DIEGO_LOSS)
    plot_network_graphs(discriminator, name)
    return inputs, discriminator


def construct_adversary_convlstm(input_size, optimizer, loss, name: str):
    inputs = Input((input_size,))
    outputs = Reshape(target_shape=(input_size, 1))(inputs)
    outputs = Conv1D(filters=2, kernel_size=2, strides=1, padding='same', activation=ACTIVATION)(outputs)
    outputs = Conv1D(filters=2, kernel_size=2, strides=1, padding='same', activation=ACTIVATION)(outputs)
    outputs = Flatten()(outputs)
    outputs = Reshape(target_shape=(input_size, 2))(outputs)
    outputs = LSTM(1, return_sequences=True, activation=ACTIVATION)(outputs)
    outputs = LSTM(1, return_sequences=True, activation=ACTIVATION)(outputs)
    outputs = LSTM(1, return_sequences=True, activation=ACTIVATION)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(4, activation=ACTIVATION)(outputs)
    outputs = Dense(2, activation=ACTIVATION)(outputs)
    outputs = Dense(1, activation=ACTIVATION)(outputs)
    discriminator = Model(inputs, outputs)
    discriminator.compile(optimizer, loss)

    discriminator.compile(DIEGO_OPT, DIEGO_LOSS)
    plot_network_graphs(discriminator, name)
    return inputs, discriminator


if __name__ == '__main__':
    main()
