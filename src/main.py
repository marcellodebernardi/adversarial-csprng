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
from utils.operation_utils import get_ith_batch, split_generator_outputs, set_trainable
from utils.input_utils import get_seed_dataset
from utils.vis_utils import *
from tqdm import tqdm
from keras import Model
from keras.layers import Input, Dense, SimpleRNN, Reshape, Flatten, Conv1D, LSTM, Lambda
from keras.activations import linear, relu, softmax
from keras.optimizers import adagrad, sgd
from keras.losses import mean_absolute_error, binary_crossentropy, mean_absolute_percentage_error
from keras.callbacks import CSVLogger
from models.activations import modulo, diagonal_max
from models.operations import drop_last_value
from models.losses import loss_discriminator, loss_predictor, loss_disc_gan, loss_pred_gan


HPC_TRAIN = False                               # set to true when training on HPC to collect data
TRAIN = [False, True]                            # Indicates whether discgan / predgan are to be trained
PRETRAIN = True                                 # if true, pretrain the discriminator/predictor
RECOMPILE = False                               # if true, models are recompiled when changing trainability
SEND_REPORT = False                             # if true, emails results to given addresses
BATCH_SIZE = 32 if HPC_TRAIN else 3          # seeds in a single batch
UNIQUE_SEEDS = 8 if HPC_TRAIN else 1         # unique seeds in each batch
BATCHES = 40 if HPC_TRAIN else 50               # batches in complete dataset
EPOCHS = 25000 if HPC_TRAIN else 100           # number of epochs for training
PRETRAIN_EPOCHS = 5000 if HPC_TRAIN else 100    # number of epochs for pre-training
ADVERSARY_MULT = 2                              # multiplier for training of the adversary
VAL_BITS = 4                                    # the number of bits of each output value or seed
MAX_VAL = 15                                   # number generated are between 0-MAX_VAL
OUTPUT_LENGTH = 100 if HPC_TRAIN else 5        # number of values generated for each seed
LEARNING_RATE = 1
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
LOG_EVERY_N = 200 if HPC_TRAIN else 10


def main():
    """ Constructs the neural networks, trains them, and logs
    all relevant information."""
    # available args: -nodisc, -nopred, -noemail
    process_cli_arguments()

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
    # define Jerry
    jerry_input = Input(shape=(1,))
    jerry_output = Dense(OUTPUT_LENGTH, activation=linear)(jerry_input)
    jerry_output = Reshape(target_shape=(5, int(OUTPUT_LENGTH / 5)))(jerry_output)
    jerry_output = SimpleRNN(int(OUTPUT_LENGTH / 5), return_sequences=True)(jerry_output)
    jerry_output = Flatten()(jerry_output)
    jerry_output = Dense(OUTPUT_LENGTH, activation=modulo(MAX_VAL))(jerry_output)
    jerry = Model(jerry_input, jerry_output, name='jerry')
    jerry.compile(UNUSED_OPT, UNUSED_LOSS)
    plot_network_graphs(jerry, 'jerry')
    utils.save_configuration(jerry, 'jerry')
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
    utils.save_configuration(diego, 'diego')
    # define the connected GAN
    discgan_output = jerry(jerry_input)
    discgan_output = diego(discgan_output)
    discgan = Model(jerry_input, discgan_output)
    discgan.compile(DISC_GAN_OPT, DISC_GAN_LOSS)
    plot_network_graphs(discgan, 'discriminative_gan')
    utils.save_configuration(discgan, 'discgan')

    # pre-train Diego
    x, y = input_utils.get_discriminator_training_dataset(jerry, BATCH_SIZE, BATCHES, OUTPUT_LENGTH, MAX_VAL)
    history = diego.fit(x, y, batch_size=BATCH_SIZE, epochs=PRETRAIN_EPOCHS, verbose=1)
    plot_pretrain_history_loss(history, '../output/plots/diego_pretrain_loss.pdf')
    print(diego.predict(x))

    # train both networks in turn
    jerry_loss, diego_loss = [], []
    # for epoch in tqdm(range(EPOCHS), desc='Train jerry and diego: '):
    #     x_d, y_d = input_utils.get_discriminator_training_dataset(jerry, BATCH_SIZE, BATCHES, OUTPUT_LENGTH, MAX_VAL)
    #     x_j, y_j = input_utils.get_jerry_training_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)
    #
    #     operation_utils.set_trainable(diego, DIEGO_OPT, DIEGO_LOSS, RECOMPILE)
    #     diego_loss.append(np.mean(diego.fit(x_d, y_d, batch_size=BATCH_SIZE, epochs=ADVERSARY_MULT, verbose=0).history['loss']))
    #     operation_utils.set_trainable(diego, DIEGO_OPT, DIEGO_LOSS, RECOMPILE, False)
    #     jerry_loss.append(discgan.fit(x_j, y_j, batch_size=BATCH_SIZE, verbose=0).history['loss'])
    #
    # vis_utils.plot_train_loss(jerry_loss, diego_loss, '../output/plots/discgan_train_loss.pdf')

    x_d, y_d = input_utils.get_discriminator_training_dataset(jerry, BATCH_SIZE, BATCHES, OUTPUT_LENGTH, MAX_VAL)
    x_j, y_j = input_utils.get_seed_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)
    operation_utils.set_trainable(diego, DIEGO_OPT, DIEGO_LOSS, True)
    print(diego.fit(x_d, y_d, batch_size=BATCH_SIZE, epochs=ADVERSARY_MULT * EPOCHS, verbose=0).history['loss'])
    operation_utils.set_trainable(diego, DIEGO_OPT, False, RECOMPILE)
    print(discgan.fit(x_j, y_j, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0).history['loss'])

    # generate outputs for one seed
    values = operation_utils.flatten_irregular_nested_iterable(jerry.predict(EVAL_SEED))
    plot_output_histogram(values, '../output/plots/discgan_jerry_output_distribution.pdf')
    plot_output_sequence(values, '../output/plots/discgan_jerry_output_sequence.pdf')
    plot_network_weights(
        operation_utils.flatten_irregular_nested_iterable(jerry.get_weights()),
        '../output/plots/jerry_weights.pdf'
    )
    utils.generate_output_file(values, jerry.name, MAX_VAL, VAL_BITS)
    # utils.log_adversary_predictions(discgan)


def predictive_gan():
    """Constructs and trains the predictive GAN consisting of
    Janice and priya."""
    # define janice
    janice_input = Input(shape=(1,))
    janice_output = Dense(OUTPUT_LENGTH, activation=linear)(janice_input)
    janice_output = Reshape(target_shape=(5, int(OUTPUT_LENGTH / 5)))(janice_output)
    janice_output = SimpleRNN(int(OUTPUT_LENGTH / 5), return_sequences=True, activation=linear)(janice_output)
    janice_output = Flatten()(janice_output)
    janice_output = Dense(OUTPUT_LENGTH, activation=modulo(MAX_VAL))(janice_output)
    janice = Model(janice_input, janice_output, name='janice')
    janice.compile(UNUSED_OPT, UNUSED_LOSS)
    plot_network_graphs(janice, 'janice')
    utils.save_configuration(janice, 'janice')
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

    # pretrain priya
    priya_x_data, priya_y_labels = split_generator_outputs(
        janice.predict(get_seed_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)[0]))
    plot_pretrain_history_loss(
        priya.fit(priya_x_data, priya_y_labels, BATCH_SIZE, PRETRAIN_EPOCHS, verbose=0),
        '../output/plots/priya_pretrain_loss.pdf')

    # main training procedure
    janice_loss, priya_loss = np.zeros(EPOCHS), np.zeros(EPOCHS)
    seed_dataset = get_seed_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)[0]
    # iterate over entire dataset
    for epoch in tqdm(range(EPOCHS), desc='Training janice and priya: '):
        priya_x_data, priya_y_labels = split_generator_outputs(janice.predict(seed_dataset))
        # iterate over portions of dataset
        for batch in range(BATCHES):
            # train predictor
            set_trainable(priya, PRIYA_OPT, PRIYA_LOSS, RECOMPILE)
            for i in range(ADVERSARY_MULT):
                priya_loss[epoch] += priya.train_on_batch(
                    get_ith_batch(priya_x_data, batch, BATCH_SIZE),
                    get_ith_batch(priya_y_labels, batch, BATCH_SIZE))
            # train generator
            set_trainable(priya, PRIYA_OPT, PRIYA_LOSS, RECOMPILE, False)
            janice_loss[epoch] += predgan.train_on_batch(
                get_ith_batch(seed_dataset, batch, BATCH_SIZE),
                get_ith_batch(priya_y_labels, batch, BATCH_SIZE))
        # update and log loss value
        janice_loss[epoch] /= BATCHES
        priya_loss[epoch] /= (BATCHES * ADVERSARY_MULT)
        if epoch % LOG_EVERY_N == 0:
            print('Janice loss: ' + str(janice_loss[epoch]) + ' / Priya loss: ' + str(priya_loss[epoch]))
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
    utils.generate_output_file(output_values, janice.name, MAX_VAL, VAL_BITS)
    # utils.log_adversary_predictions(predgan)


def process_cli_arguments():
    global TRAIN
    global SEND_REPORT

    if "-help" in sys.argv or "-h" in sys.argv:
        print("Optional arguments include:\n"
              + "-nodisc\t\tdo not train discriminator\n"
              + "-nopred\t\tdo not train predictor\n"
              + "-noemail\tdo not report by email\n")
        exit(0)

    if '-nodisc' in sys.argv:
        TRAIN[0] = False

    if '-nopred' in sys.argv:
        TRAIN[1] = False

    if '-noemail' in sys.argv:
        SEND_REPORT = False


if __name__ == '__main__':
    main()
