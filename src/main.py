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
and Pramesh (the predictor). Janice produces sequences of real
numbers in the same fashion as Jerry. Pramesh receives as
input the entire sequence produced by Janice, except for the
last value, which it attempts to predict.

The main function defines these networks and trains them.
"""

import sys
import numpy as np
from utils import utils, vis_utils, input_utils, operation_utils
from tqdm import tqdm
from keras import Model
from keras.layers import Input, Dense, SimpleRNN, Reshape, Flatten, Conv1D, LSTM, Lambda
from keras.activations import linear, softmax, relu
from keras.optimizers import adagrad
from models.activations import modulo
from models.operations import drop_last_value
from models.losses import loss_discriminator, loss_predictor, loss_adv
from evaluators import pnb


HPC_TRAIN = False                               # set to true when training on HPC to collect data
PRETRAIN = True                                 # if true, pretrain the discriminator/predictor
BATCH_SIZE = 4096 if HPC_TRAIN else 2           # seeds in a single batch
UNIQUE_SEEDS = 128 if HPC_TRAIN else 2          # unique seeds in each batch
BATCHES = 50 if HPC_TRAIN else 1                # batches in complete dataset
EPOCHS = 300000 if HPC_TRAIN else 10            # number of epochs for training
PRETRAIN_EPOCHS = 15000 if HPC_TRAIN else 5     # number of epochs for pre-training
ADVERSARY_MULT = 2                              # multiplier for training of the adversary
MAX_VAL = 127 if HPC_TRAIN else 20              # number generated are between 0-MAX_VAL
OUTPUT_LENGTH = 5000 if HPC_TRAIN else 10       # number of values generated for each seed
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
    diego.fit(x, y, BATCH_SIZE, PRETRAIN_EPOCHS, verbose=0)

    # train both networks in turn
    for epoch in tqdm(range(EPOCHS), desc='Train: '):
        # todo currently each epoch only alternates between jerry and diego once
        # todo should alternate at each batch
        # train diego
        x, y = input_utils.get_discriminator_training_dataset(jerry, BATCH_SIZE, BATCHES, OUTPUT_LENGTH, MAX_VAL)
        operation_utils.set_trainable(diego)
        for iteration in range(ADVERSARY_MULT):
            diego.fit(x, y, BATCH_SIZE, verbose=0)
        # train jerry
        x, y = input_utils.get_jerry_training_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)
        operation_utils.set_trainable(diego, False)
        discgan.fit(x, y, verbose=0)
    # generate output file for one seed
    utils.generate_output_file(jerry, MAX_VAL)
    pnb.evaluate('../sequences/jerry.txt')


def predictive_gan():
    """Constructs and trains the predictive GAN consisting of
    Janice and Pramesh."""
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
    # define pramesh
    pramesh_input = Input(shape=(OUTPUT_LENGTH - 1,))
    pramesh_output = Dense(OUTPUT_LENGTH)(pramesh_input)
    pramesh_output = Reshape(target_shape=(5, int(OUTPUT_LENGTH / 5)))(pramesh_output)
    pramesh_output = LSTM(int(OUTPUT_LENGTH / 5), return_sequences=True, activation=linear)(pramesh_output)
    pramesh_output = Flatten()(pramesh_output)
    pramesh_output = Dense(1, activation=relu)(pramesh_output)
    pramesh = Model(pramesh_input, pramesh_output)
    pramesh.compile(adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE), loss_predictor(MAX_VAL))
    vis_utils.plot_network_graphs(pramesh, 'pramesh')
    # connect GAN
    output_predgan = janice(janice_input)
    output_predgan = Lambda(
        drop_last_value(OUTPUT_LENGTH, BATCH_SIZE),
        name='adversarial_drop_last_value')(output_predgan)
    output_predgan = pramesh(output_predgan)
    predgan = Model(janice_input, output_predgan, name='predictive_gan')
    predgan.compile(adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE), loss_adv(loss_predictor(MAX_VAL)))
    vis_utils.plot_network_graphs(predgan, 'predictive_gan')

    # pretrain pramesh
    seed_dataset = input_utils.get_jerry_training_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)[0]
    for epoch in tqdm(range(PRETRAIN_EPOCHS), desc='Pre-training pramesh ...'):
        for janice_input in seed_dataset:
            janice_output = janice.predict(np.array([janice_input]))
            pramesh_input, pramesh_output = operation_utils.split_generator_output(janice_output, 1)
            pramesh.fit(pramesh_input, pramesh_output, verbose=0)

    # train both janice and pramesh
    seed_dataset = input_utils.get_jerry_training_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)[0]
    for epoch in tqdm(range(EPOCHS), desc='Train janice and pramesh: '):
        for janice_input in seed_dataset:
            janice_output = janice.predict(np.array([janice_input]))
            pramesh_input, pramesh_output = operation_utils.split_generator_output(janice_output, 1)
            # train predictor
            operation_utils.set_trainable(pramesh)
            for i in range(ADVERSARY_MULT):
                pramesh.fit(pramesh_input, pramesh_output, verbose=0)
            # train generator
            operation_utils.set_trainable(pramesh, False)
            predgan.fit(np.array([janice_input]), pramesh_output, verbose=0)

    utils.generate_output_file(janice, MAX_VAL)
    pnb.evaluate('../sequences/' + str(janice.name) + '.txt')


if __name__ == '__main__':
    main()
