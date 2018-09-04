# Marcello De Bernardi, University of Oxford
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
"""

import sys
import math
import tensorflow as tf
from components.models import GAN, PredGAN
from components import inputs, losses
from utils import files

# main settings
NUMBER_OF_BITS_PRODUCED = 1000000000
# hyper-parameter with list of options for robustness testing
INPUT_SIZE = {'default': 2, 'options': [1, 2, 3, 4, 5]}
GEN_WIDTH = {'default': 32, 'options': [8, 16, 32, 64, 128, 256, 512, 1024]}
OUTPUT_SIZE = {'default': 8, 'options': [8, 16, 32, 64, 128, 256, 512, 1024]}
OUTPUT_BITS = {'default': 16, 'options': [2, 4, 8, 16, 32]}
MAX_VAL = {'default': 65535, 'options': [(2 ** i) - 1 for i in OUTPUT_BITS['options']]}
BATCH_SIZE = {'default': 2048, 'options': [8, 16, 32, 64, 128, 256, 512, 1024, 2048]}
LEARNING_RATE = {'default': 0.010, 'options': [0.001, 0.010, 0.020, 0.040, 0.080, 0.16, 0.32, 0.64]}
D_TYPE = tf.float64
# optimizers and losses
GEN_OPT = tf.train.AdamOptimizer
OPP_OPT = tf.train.AdamOptimizer
BETA1 = 0.9999
BETA2 = 0.9999
# training settings
STEPS = [10, 100, 1000, 10000, 100000]
EXPERIMENT_REPETITIONS = 10
EVAL_MILESTONES = 5
ADV_MULT = 3
EVAL_SEED = 10

# data = number of batches * batch size * output vector size * digits per number
PLOT_DIR = '../output/plots/'
DATA_DIR = '../output/data/'
SEQN_DIR = '../output/sequences/'
GRAPH_DIR = '../output/tensor_board/'


def main():
    """ Constructs the neural networks, trains them, and logs all relevant information. """

    if sys.argv[1] == 'gan':
        print('# GAN - batch size: ', BATCH_SIZE)
        run_experiment(run_discgan)
    elif sys.argv[1] == 'predgan':
        print('# PredGAN - batch size: ', BATCH_SIZE)
        run_experiment(run_predgan)


def run_discgan(input_size, gen_width, output_size, max_val, batch_size, gen_opt, opp_opt, steps, eval_data, folder):
    """ Constructs, trains and evaluates the discriminative GAN with the provided experiment parameters.
    :param input_size: the dimensionality of the generator input
    :param gen_width: the width of the generator's hidden layers
    :param output_size: the dimensionality of the generator output
    :param max_val: the upper bound for the values produced by the generator
    :param batch_size: size of batches on which gradient update is performed
    :param gen_opt: optimizer used by the generator
    :param opp_opt: optimizer used by the discriminator
    :param steps: number of times the generator/discriminator pair are updated
    :param eval_data: evaluation dataset used for the generator
    :param folder: subdirectory for experiment run with current parameters """

    gan = GAN(input_size, gen_width, output_size, max_val, gen_width, ADV_MULT) \
        .with_distributions(inputs.noise_prior_tf, inputs.reference_distribution_tf) \
        .with_optimizers(gen_opt, opp_opt) \
        .with_loss_functions(losses.generator_classification_loss, losses.discriminator_classification_loss)

    for i in range(EXPERIMENT_REPETITIONS):
        for j in range(EVAL_MILESTONES):
            gan.train(batch_size, steps)
            eval_out = gan.predict(eval_data, batch_size)
            files.write_numbers_to_ascii_file(eval_out,
                                              SEQN_DIR + 'discriminative/' + folder
                                              + 'steps-' + str(steps)
                                              + '-exp-' + str(i)
                                              + '-mile-' + str(j)
                                              + '-gan.txt')
    # produce output
    files.write_to_file(gan.get_recorded_losses()['generator'], PLOT_DIR + '/generator_loss.txt')
    files.write_to_file(gan.get_recorded_losses()['discriminator'], PLOT_DIR + '/discriminator_loss.txt')


def run_predgan(input_size, gen_width, output_size, max_val, batch_size, gen_opt, opp_opt, steps, eval_data, folder):
    """ Constructs, trains and evaluates the predictive GAN with the provided experiment parameters.
    :param input_size: the dimensionality of the generator input
    :param gen_width: the width of the generator's hidden layers
    :param output_size: the dimensionality of the generator output
    :param max_val: the upper bound for the values produced by the generator
    :param batch_size: size of batches on which gradient update is performed
    :param gen_opt: optimizer used by the generator
    :param opp_opt: optimizer used by the predictor
    :param steps: number of times the generator/predictor pair are updated
    :param eval_data: evaluation dataset used for the generator
    :param folder: subdirectory for experiment run with current parameters """

    predgan = PredGAN(input_size, gen_width, output_size, max_val, gen_width, ADV_MULT) \
        .with_distributions(inputs.noise_prior_tf) \
        .with_optimizers(gen_opt, opp_opt) \
        .with_loss_functions(losses.build_generator_regression_loss(max_val),
                             losses.build_predictor_regression_loss(max_val))
    for i in range(EXPERIMENT_REPETITIONS):
        for j in range(EVAL_MILESTONES):
            predgan.train(batch_size, steps)
            eval_out = predgan.predict(eval_data, batch_size)
            files.write_numbers_to_ascii_file(eval_out,
                                              SEQN_DIR + 'predictive/' + folder
                                              + 'steps-' + str(steps)
                                              + '-exp-' + str(i)
                                              + '-mile-' + str(j)
                                              + '-predgan.txt')
    # produce output
    files.write_to_file(predgan.get_recorded_losses()['generator'], PLOT_DIR + '/generator_loss.txt')
    files.write_to_file(predgan.get_recorded_losses()['discriminator'], PLOT_DIR + '/discriminator_loss.txt')


def run_experiment(train_and_eval):
    """ Calls the training and evaluation procedure repeatedly, iterating
    over the possible modified values. """
    steps = STEPS[0]
    # defaults
    num_of_elements = NUMBER_OF_BITS_PRODUCED / (OUTPUT_SIZE['default'] * OUTPUT_BITS['default'])
    eval_data = inputs.get_eval_input_numpy(EVAL_SEED, num_of_elements, INPUT_SIZE['default'])
    folder = 'defaults/'
    train_and_eval(INPUT_SIZE['default'], GEN_WIDTH['default'], OUTPUT_SIZE['default'], MAX_VAL['default'],
                   BATCH_SIZE['default'], GEN_OPT(LEARNING_RATE['default'], BETA1, BETA2),
                   OPP_OPT(LEARNING_RATE['default'], BETA1, BETA2), steps, eval_data, folder)
    # vary input size
    for input_size in INPUT_SIZE['options']:
        num_of_elements = NUMBER_OF_BITS_PRODUCED / (OUTPUT_SIZE['default'] * OUTPUT_BITS['default'])
        eval_data = inputs.get_eval_input_numpy(EVAL_SEED, num_of_elements, input_size)
        folder = 'input_size/' + str(input_size) + '/'
        train_and_eval(input_size, GEN_WIDTH['default'], OUTPUT_SIZE['default'], MAX_VAL['default'],
                       BATCH_SIZE['default'], GEN_OPT(LEARNING_RATE['default'], BETA1, BETA2),
                       OPP_OPT(LEARNING_RATE['default'], BETA1, BETA2), steps, eval_data, folder)
    # vary gen width
    num_of_elements = NUMBER_OF_BITS_PRODUCED / (OUTPUT_SIZE['default'] * OUTPUT_BITS['default'])
    eval_data = inputs.get_eval_input_numpy(EVAL_SEED, num_of_elements, INPUT_SIZE['default'])
    for gen_width in GEN_WIDTH['options']:
        folder = 'num_of_elements/' + str(num_of_elements) + '/'
        train_and_eval(INPUT_SIZE['default'], gen_width, OUTPUT_SIZE['default'], MAX_VAL['default'],
                       BATCH_SIZE['default'], GEN_OPT(LEARNING_RATE['default'], BETA1, BETA2),
                       OPP_OPT(LEARNING_RATE['default'], BETA1, BETA2), steps, eval_data, folder)
    # vary gen output size
    for output_size in OUTPUT_SIZE['options']:
        num_of_elements = NUMBER_OF_BITS_PRODUCED / (output_size * OUTPUT_BITS['default'])
        eval_data = inputs.get_eval_input_numpy(EVAL_SEED, num_of_elements, INPUT_SIZE['default'])
        folder = 'output_size/' + str(output_size) + '/'
        train_and_eval(INPUT_SIZE['default'], GEN_WIDTH['default'], output_size, MAX_VAL['default'],
                       BATCH_SIZE['default'], GEN_OPT(LEARNING_RATE['default'], BETA1, BETA2),
                       OPP_OPT(LEARNING_RATE['default'], BETA1, BETA2), steps, eval_data, folder)
    # vary gen max value
    for max_val in MAX_VAL['options']:
        num_of_elements = NUMBER_OF_BITS_PRODUCED / (OUTPUT_SIZE['default'] * round(math.log(max_val, 2)))
        eval_data = inputs.get_eval_input_numpy(EVAL_SEED, num_of_elements, INPUT_SIZE['default'])
        folder = 'max_val/' + str(max_val) + '/'
        train_and_eval(INPUT_SIZE['default'], GEN_WIDTH['default'], OUTPUT_SIZE['default'], max_val,
                       BATCH_SIZE['default'], GEN_OPT(LEARNING_RATE['default'], BETA1, BETA2),
                       OPP_OPT(LEARNING_RATE['default'], BETA1, BETA2), steps, eval_data, folder)
    # vary batch size
    num_of_elements = NUMBER_OF_BITS_PRODUCED / (OUTPUT_SIZE['default'] * OUTPUT_BITS['default'])
    eval_data = inputs.get_eval_input_numpy(EVAL_SEED, num_of_elements, INPUT_SIZE['default'])
    for batch_size in BATCH_SIZE['options']:
        folder = 'batch_size/' + str(batch_size) + '/'
        train_and_eval(INPUT_SIZE['default'], GEN_WIDTH['default'], OUTPUT_SIZE['default'], MAX_VAL['default'],
                       batch_size, GEN_OPT(LEARNING_RATE['default'], BETA1, BETA2),
                       OPP_OPT(LEARNING_RATE['default'], BETA1, BETA2), steps, eval_data, folder)
    # vary learning rate
    num_of_elements = NUMBER_OF_BITS_PRODUCED / (OUTPUT_SIZE['default'] * OUTPUT_BITS['default'])
    eval_data = inputs.get_eval_input_numpy(EVAL_SEED, num_of_elements, INPUT_SIZE['default'])
    for learning_rate in LEARNING_RATE['options']:
        folder = 'learning_rate/' + str(learning_rate) + '/'
        train_and_eval(INPUT_SIZE['default'], GEN_WIDTH['default'], OUTPUT_SIZE['default'], MAX_VAL['default'],
                       BATCH_SIZE['default'], GEN_OPT(learning_rate, BETA1, BETA2),
                       OPP_OPT(learning_rate, BETA1, BETA2), steps, eval_data, folder)


if __name__ == '__main__':
    main()
