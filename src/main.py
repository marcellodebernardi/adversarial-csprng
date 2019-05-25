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
from experiment import Experiment, ExperimentFunction
from components.models import GAN, PredGAN
from components import inputs, losses
from utils import files

# training settings
STEPS = [10, 100, 1000, 10000, 100000]
EXPERIMENT_REPETITIONS = 10
EVAL_MILESTONES = 5
EVAL_SEED = 10
PLOT_DIR = '../output/plots/'
DATA_DIR = '../output/data/'
SEQN_DIR = '../output/sequences/'
GRAPH_DIR = '../output/tensor_board/'


def main():
    """ Constructs the neural networks, trains them, and logs all relevant information. """

    if sys.argv[1] == 'gan':
        Experiment(DiscganExperimentFunction(), EXPERIMENT_REPETITIONS).perform(STEPS[0])
        Experiment(DiscganExperimentFunction(), EXPERIMENT_REPETITIONS, 'input_size').perform(STEPS[0])
        Experiment(DiscganExperimentFunction(), EXPERIMENT_REPETITIONS, 'gen_width').perform(STEPS[0])
        Experiment(DiscganExperimentFunction(), EXPERIMENT_REPETITIONS, 'output_size').perform(STEPS[0])
        Experiment(DiscganExperimentFunction(), EXPERIMENT_REPETITIONS, 'max_val').perform(STEPS[0])
        Experiment(DiscganExperimentFunction(), EXPERIMENT_REPETITIONS, 'batch_size').perform(STEPS[0])
        Experiment(DiscganExperimentFunction(), EXPERIMENT_REPETITIONS, 'learning_rate').perform(STEPS[0])
    elif sys.argv[1] == 'predgan':
        Experiment(PredganExperimentFunction(), EXPERIMENT_REPETITIONS).perform(STEPS[0])
        Experiment(PredganExperimentFunction(), EXPERIMENT_REPETITIONS, 'input_size').perform(STEPS[0])
        Experiment(PredganExperimentFunction(), EXPERIMENT_REPETITIONS, 'gen_width').perform(STEPS[0])
        Experiment(PredganExperimentFunction(), EXPERIMENT_REPETITIONS, 'output_size').perform(STEPS[0])
        Experiment(PredganExperimentFunction(), EXPERIMENT_REPETITIONS, 'max_val').perform(STEPS[0])
        Experiment(PredganExperimentFunction(), EXPERIMENT_REPETITIONS, 'batch_size').perform(STEPS[0])
        Experiment(PredganExperimentFunction(), EXPERIMENT_REPETITIONS, 'learning_rate').perform(STEPS[0])


class DiscganExperimentFunction(ExperimentFunction):
    def run_function(self, input_size, gen_width, output_size, max_val, batch_size, learning_rate, adv_mul, steps,
                     eval_data, folder):
        """ Constructs, trains and evaluates the discriminative GAN with the provided experiment parameters.
        :param input_size: the dimensionality of the generator input
        :param gen_width: the width of the generator's hidden layers
        :param output_size: the dimensionality of the generator output
        :param max_val: the upper bound for the values produced by the generator
        :param batch_size: size of batches on which gradient update is performed
        :param learning_rate: optimizer learning rate
        :param adv_mul: the gradient update multiplier for the adversary
        :param steps: number of times the generator/discriminator pair are updated
        :param eval_data: evaluation dataset used for the generator
        :param folder: subdirectory for experiment run with current parameters """

        gan = GAN(input_size, gen_width, output_size, max_val, gen_width, adv_mul) \
            .with_distributions(inputs.noise_prior_tf, inputs.reference_distribution_tf) \
            .with_optimizers(tf.train.AdamOptimizer(learning_rate, 0.9999, 0.9999),
                             tf.train.AdamOptimizer(learning_rate, 0.9999, 0.9999)) \
            .with_loss_functions(losses.generator_classification_loss, losses.discriminator_classification_loss)

        for j in range(EVAL_MILESTONES):
            gan.train(batch_size, steps)
            eval_out = gan.predict(eval_data, batch_size)
            files.write_numbers_to_ascii_file(eval_out,
                                              SEQN_DIR + 'discriminative/' + folder
                                              + 'steps-' + str(steps)
                                              + '-mile-' + str(j)
                                              + '-gan.txt')
        # produce output
        files.write_to_file(gan.get_recorded_losses()['generator'], PLOT_DIR + '/generator_loss.txt')
        files.write_to_file(gan.get_recorded_losses()['discriminator'], PLOT_DIR + '/discriminator_loss.txt')


class PredganExperimentFunction(ExperimentFunction):
    def run_function(self, input_size, gen_width, output_size, max_val, batch_size, learning_rate, adv_mul, steps,
                     eval_data, folder):
        """ Constructs, trains and evaluates the predictive GAN with the provided experiment parameters.
        :param input_size: the dimensionality of the generator input
        :param gen_width: the width of the generator's hidden layers
        :param output_size: the dimensionality of the generator output
        :param max_val: the upper bound for the values produced by the generator
        :param batch_size: size of batches on which gradient update is performed
        :param learning_rate: optimizer learning rate
        :param adv_mul: the gradient update multiplier for the adversary
        :param steps: number of times the generator/predictor pair are updated
        :param eval_data: evaluation dataset used for the generator
        :param folder: subdirectory for experiment run with current parameters """

        predgan = PredGAN(input_size, gen_width, output_size, max_val, gen_width, adv_mul) \
            .with_distributions(inputs.noise_prior_tf) \
            .with_optimizers(tf.train.AdamOptimizer(learning_rate, 0.9999, 0.9999),
                             tf.train.AdamOptimizer(learning_rate, 0.9999, 0.9999)) \
            .with_loss_functions(losses.build_generator_regression_loss(max_val),
                                 losses.build_predictor_regression_loss(max_val))
        for j in range(EVAL_MILESTONES):
            predgan.train(batch_size, steps)
            eval_out = predgan.predict(eval_data, batch_size)
            files.write_numbers_to_ascii_file(eval_out,
                                              SEQN_DIR + 'predictive/' + folder
                                              + 'steps-' + str(steps)
                                              + '-mile-' + str(j)
                                              + '-predgan.txt')
        # produce output
        files.write_to_file(predgan.get_recorded_losses()['generator'], PLOT_DIR + '/generator_loss.txt')
        files.write_to_file(predgan.get_recorded_losses()['discriminator'], PLOT_DIR + '/discriminator_loss.txt')


if __name__ == '__main__':
    main()
