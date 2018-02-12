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
from utils import utils, vis_utils, input_utils, operation_utils
from utils.operation_utils import get_ith_batch
from tqdm import tqdm
from keras import Model
from keras.layers import Input, Dense, SimpleRNN, Reshape, Flatten, Conv1D, LSTM, Lambda
from keras.activations import linear, relu
from keras.optimizers import adagrad, sgd
from models.activations import modulo, diagonal_max
from models.operations import drop_last_value
from models.losses import loss_discriminator, loss_predictor, loss_disc_gan, loss_pred_gan


HPC_TRAIN = False                               # set to true when training on HPC to collect data
TRAIN = [True, True]                            # Indicates whether discgan / predgan are to be trained
PRETRAIN = True                                 # if true, pretrain the discriminator/predictor
RECOMPILE = False                               # if true, models are recompiled when changing trainability
SEND_REPORT = False                             # if true, emails results to given addresses
BATCH_SIZE = 4096 if HPC_TRAIN else 10          # seeds in a single batch
UNIQUE_SEEDS = 128 if HPC_TRAIN else 10         # unique seeds in each batch
BATCHES = 50 if HPC_TRAIN else 10               # batches in complete dataset
EPOCHS = 300000 if HPC_TRAIN else 100           # number of epochs for training
PRETRAIN_EPOCHS = 15000 if HPC_TRAIN else 160    # number of epochs for pre-training
ADVERSARY_MULT = 2                              # multiplier for training of the adversary
VAL_BITS = 8                                    # the number of bits of each output value or seed
MAX_VAL = 255                                   # number generated are between 0-MAX_VAL
OUTPUT_LENGTH = 5000 if HPC_TRAIN else 5        # number of values generated for each seed
LEARNING_RATE = 1
CLIP_VALUE = 0.5

# losses and optimizers
DIEGO_OPT = adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE)
DIEGO_LOSS = loss_discriminator
DISC_GAN_OPT = sgd(lr=LEARNING_RATE, clipvalue=CLIP_VALUE)
DISC_GAN_LOSS = loss_disc_gan
PRIYA_OPT = adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE)
PRIYA_LOSS = loss_predictor(MAX_VAL)
PRED_GAN_OPT = adagrad(lr=LEARNING_RATE, clipvalue=CLIP_VALUE)
PRED_GAN_LOSS = loss_pred_gan(MAX_VAL)
UNUSED_OPT = 'adagrad'
UNUSED_LOSS = 'binary_crossentropy'

# evaluation seed
EVAL_SEED = input_utils.get_random_sequence(1, MAX_VAL)


def main():
    """ Constructs the neural networks, trains them, and logs
    all relevant information."""
    # available args: -t, -nodisc, -nopred, -noemail
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
    vis_utils.plot_network_graphs(jerry, 'jerry')
    utils.save_configuration(jerry, 'jerry')
    # define Diego
    diego_input = Input(shape=(OUTPUT_LENGTH,))
    diego_output = Dense(OUTPUT_LENGTH)(diego_input)
    diego_output = Reshape(target_shape=(5, int(OUTPUT_LENGTH / 5)))(diego_output)
    diego_output = Conv1D(int(OUTPUT_LENGTH / 4), 4)(diego_output)
    diego_output = Flatten()(diego_output)
    diego_output = Dense(int(OUTPUT_LENGTH / 4), activation=diagonal_max(100))(diego_output)
    diego_output = Dense(1)(diego_output)
    diego = Model(diego_input, diego_output)
    diego.compile(DIEGO_OPT, DIEGO_LOSS)
    vis_utils.plot_network_graphs(diego, 'diego')
    utils.save_configuration(diego, 'diego')
    # define the connected GAN
    discgan_output = jerry(jerry_input)
    discgan_output = diego(discgan_output)
    discgan = Model(jerry_input, discgan_output)
    discgan.compile(DISC_GAN_OPT, DISC_GAN_LOSS)
    vis_utils.plot_network_graphs(discgan, 'discriminative_gan')
    utils.save_configuration(discgan, 'discgan')

    # pre-train Diego
    x, y = input_utils.get_discriminator_training_dataset(jerry, BATCH_SIZE, BATCHES, OUTPUT_LENGTH, MAX_VAL)
    history = diego.fit(x, y, batch_size=BATCH_SIZE, epochs=PRETRAIN_EPOCHS, verbose=0)
    vis_utils.plot_pretrain_history_loss(history, '../output/plots/diego_pretrain_loss.pdf')

    # train both networks in turn
    jerry_loss, diego_loss = [], []
    for epoch in tqdm(range(EPOCHS), desc='Train jerry and diego: '):
        x_d, y_d = input_utils.get_discriminator_training_dataset(jerry, BATCH_SIZE, BATCHES, OUTPUT_LENGTH, MAX_VAL)
        x_j, y_j = input_utils.get_jerry_training_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)
        jerry_l, diego_l = 0, 0
        for batch in range(BATCHES):
            # train diego
            operation_utils.set_trainable(diego, DIEGO_OPT, DIEGO_LOSS, RECOMPILE)
            for iteration in range(ADVERSARY_MULT):
                diego_l += diego.train_on_batch(get_ith_batch(x_d, batch, BATCH_SIZE), get_ith_batch(y_d, batch, BATCH_SIZE))
            # train jerry
            operation_utils.set_trainable(diego, DIEGO_OPT, DIEGO_LOSS, RECOMPILE, False)
            jerry_l += discgan.train_on_batch(get_ith_batch(x_j, batch, BATCH_SIZE), get_ith_batch(y_j, batch, BATCH_SIZE))
        jerry_loss.append(jerry_l/BATCHES)
        diego_loss.append(diego_l/(BATCHES * ADVERSARY_MULT))
    vis_utils.plot_train_loss(jerry_loss, diego_loss, '../output/plots/discgan_train_loss.pdf')
    # generate output file for one seed
    vis_utils.plot_output_histogram(
        jerry.predict(EVAL_SEED),
        '../output/plots/discgan_jerry_output_distribution.pdf')
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
    janice.compile(UNUSED_OPT, UNUSED_LOSS)
    vis_utils.plot_network_graphs(janice, 'janice')
    utils.save_configuration(janice, 'janice')
    # define priya
    priya_input = Input(shape=(OUTPUT_LENGTH - 1,))
    priya_output = Dense(OUTPUT_LENGTH)(priya_input)
    priya_output = Reshape(target_shape=(5, int(OUTPUT_LENGTH / 5)))(priya_output)
    priya_output = LSTM(int(OUTPUT_LENGTH / 5), return_sequences=True, activation=linear)(priya_output)
    priya_output = Flatten()(priya_output)
    priya_output = Dense(1, activation=relu)(priya_output)
    priya = Model(priya_input, priya_output)
    priya.compile(PRIYA_OPT, PRIYA_LOSS)
    vis_utils.plot_network_graphs(priya, 'priya')
    utils.save_configuration(priya, 'priya')
    # connect GAN
    output_predgan = janice(janice_input)
    output_predgan = Lambda(
        drop_last_value(OUTPUT_LENGTH, BATCH_SIZE),
        name='adversarial_drop_last_value')(output_predgan)
    output_predgan = priya(output_predgan)
    predgan = Model(janice_input, output_predgan, name='predictive_gan')
    predgan.compile(PRED_GAN_OPT, PRED_GAN_LOSS)
    vis_utils.plot_network_graphs(predgan, 'predictive_gan')
    # utils.save_configuration(predgan, 'predgan')

    # pretrain priya
    seed_dataset = input_utils.get_jerry_training_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)[0]
    janice_output = janice.predict(seed_dataset)
    priya_input, priya_output = operation_utils.split_generator_outputs_batch(janice_output, 1)
    history = priya.fit(priya_input, priya_output, batch_size=BATCH_SIZE, epochs=PRETRAIN_EPOCHS, verbose=0)
    vis_utils.plot_pretrain_history_loss(history, '../output/plots/priya_pretrain_loss.pdf')

    # train both janice and priya
    janice_loss, priya_loss = [], []
    for epoch in tqdm(range(EPOCHS), desc='Train janice and priya: '):
        seed_dataset = input_utils.get_jerry_training_dataset(BATCH_SIZE, BATCHES, UNIQUE_SEEDS, MAX_VAL)[0]
        janice_l, priya_l = 0, 0
        for batch in range(BATCHES):
            janice_output = janice.predict_on_batch(get_ith_batch(seed_dataset, batch, BATCH_SIZE))
            # priya_input, priya_output = operation_utils.split_generator_output(janice_output, 1)
            priya_input, priya_output = operation_utils.split_generator_outputs_batch(janice_output, 1)
            operation_utils.set_trainable(priya, PRIYA_OPT, PRIYA_LOSS, RECOMPILE)
            for i in range(ADVERSARY_MULT):
                priya_l += priya.train_on_batch(priya_input, priya_output)
            # train generator
            operation_utils.set_trainable(priya, PRIYA_OPT, PRIYA_LOSS, RECOMPILE, False)
            janice_l += predgan.train_on_batch(get_ith_batch(seed_dataset, batch, BATCH_SIZE), priya_output)
        janice_loss.append(janice_l / BATCHES)
        priya_loss.append(priya_l / (BATCHES * ADVERSARY_MULT))

    vis_utils.plot_train_loss(janice_loss, priya_loss, '../output/plots/predgan_train_loss.pdf')
    vis_utils.plot_output_histogram(
        janice.predict(EVAL_SEED),
        '../output/plots/predgan_janice_output_distribution.pdf')
    utils.generate_output_file(janice, MAX_VAL, VAL_BITS)
    # pnb.evaluate('../sequences/' + str(janice.name) + '.txt')


def process_cli_arguments():
    global TESTING
    global TRAIN
    global SEND_REPORT

    if "-help" in sys.argv or "-h" in sys.argv:
        print("Optional arguments include:\n" + "-t\t\ttesting mode\n"
              + "-nodisc\t\tdo not train discriminator\n"
              + "-nopred\t\tdo not train predictor\n"
              + "-noemail\tdo not report by email\n")
        exit(0)

    if "-t" in sys.argv:
        TESTING = True

    if '-nodisc' in sys.argv:
        TRAIN[0] = False

    if '-nopred' in sys.argv:
        TRAIN[1] = False

    if '-noemail' in sys.argv:
        SEND_REPORT = False


if __name__ == '__main__':
    main()
