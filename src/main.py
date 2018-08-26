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
-nodisc         SKIP DISCRIMINATIVE GAN: does not train discriminative GAN
-nopred         SKIP PREDICTIVE GAN: does not train predictive GAN
"""

import sys
import tensorflow as tf
from components.models import GAN, PredGAN
from components import inputs, losses
from utils import files

# main settings
HPC_TRAIN = '-t' not in sys.argv  # set to true when training on HPC to collect data
LEARN_LEVEL = 2 if '-highlr' in sys.argv else 0 if '-lowlr' in sys.argv else 1

# hyper-parameters
OUTPUT_SIZE = 8
MAX_VAL = 65535
OUTPUT_BITS = 16
BATCH_SIZE = 2046 if HPC_TRAIN else 10
LEARNING_RATE = {2: 0.1, 1: 0.02, 0: 0.008}[LEARN_LEVEL]
GEN_WIDTH = 30 if HPC_TRAIN else 10
DATA_TYPE = tf.float64

# optimizers and losses
GEN_OPT = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.9999, beta2=0.9999)
OPP_OPT = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.9999, beta2=0.9999)

# training settings
TRAIN = ['-nodisc' not in sys.argv, '-nopred' not in sys.argv]
STEPS = 1000000 if '-long' in sys.argv else 10000 if '-short' in sys.argv else 200000 if HPC_TRAIN else 40
PRE_STEPS = 100 if HPC_TRAIN else 5
ADV_MULT = 3

# logging and evaluation
EVAL_BATCHES = 400 if HPC_TRAIN else 10
EVAL_DATA = inputs.get_eval_input_numpy(10, EVAL_BATCHES, BATCH_SIZE)
LOG_EVERY_N = 10 if HPC_TRAIN else 1
PLOT_DIR = '../output/plots/'
DATA_DIR = '../output/data/'
SEQN_DIR = '../output/sequences/'
GRAPH_DIR = '../output/model_graphs/'


def main():
    """ Constructs the neural networks, trains them, and logs
        all relevant information.
    """
    # train discriminative GAN
    if TRAIN[0]:
        print('# GAN - batch size: ', BATCH_SIZE)
        run_discgan()
    # train predictive GAN
    if TRAIN[1]:
        print('# PredGAN - batch size: ', BATCH_SIZE)
        run_predgan()

    # print used settings for convenience
    print('\n[COMPLETE] (DISCGAN: ' + str(TRAIN[0]) + ', PREDGAN: ' + str(TRAIN[1]) + ')')


def run_discgan():
    """ Constructs and trains the discriminative GAN consisting of
        Jerry and Diego.
    """
    # code follows the examples from
    # https://github.com/tensorflow/models/blob/master/research/gan/tutorial.ipynb

    gan = GAN(input_width=2, gen_width=30, gen_out_width=8, disc_width=30, adv_multiplier=3)\
        .with_distributions(inputs.noise_prior_tf, inputs.reference_distribution_tf)\
        .with_optimizers(GEN_OPT, OPP_OPT)\
        .with_loss_functions(losses.generator_classification_loss, losses.discriminator_classification_loss)

    for i in range(10):
        print('GAN: training section ' + str(i) + ' -----')
        try:
            gan.train(BATCH_SIZE, 100)
            eval_out = gan.predict(EVAL_DATA, BATCH_SIZE)
            files.write_output_file(eval_out, SEQN_DIR + str(i) + '_' + 'gan.txt')
        except KeyboardInterrupt:
            print('[INTERRUPTED BY USER]')

        # produce output
        files.write_to_file(gan.get_recorded_losses()['generator'], PLOT_DIR + '/generator_loss.txt')
        files.write_to_file(gan.get_recorded_losses()['discriminator'], PLOT_DIR + '/discriminator_loss.txt')


def run_predgan():
    """ Constructs, trains and evaluates the predictive GAN. """
    predgan = PredGAN(input_width=2, gen_width=30, gen_out_width=8, disc_width=30, adv_multiplier=3) \
        .with_distributions(inputs.noise_prior_tf) \
        .with_optimizers(GEN_OPT, OPP_OPT) \
        .with_loss_functions(losses.build_generator_regression_loss(MAX_VAL),
                             losses.build_predictor_regression_loss(MAX_VAL))

    for i in range(10):
        print('GAN: training section ' + str(i) + ' -----')
        try:
            predgan.train(BATCH_SIZE, 100)
            eval_out = predgan.predict(EVAL_DATA, BATCH_SIZE)
            files.write_output_file(eval_out, SEQN_DIR + str(i) + '_' + 'predgan.txt')
        except KeyboardInterrupt:
            print('[INTERRUPTED BY USER]')

        # produce output
        files.write_to_file(predgan.get_recorded_losses()['generator'], PLOT_DIR + '/generator_loss.txt')
        files.write_to_file(predgan.get_recorded_losses()['discriminator'], PLOT_DIR + '/discriminator_loss.txt')

    # janice_input_t = tf.placeholder(shape=[BATCH_SIZE, 2], dtype=tf.float32)
    # janice_output_t = generator(janice_input_t)
    # janice_true_t = tf.strided_slice(janice_output_t, [0, -0], [BATCH_SIZE, 1], [1, 1])
    # priya_pred_t = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32)
    # # priya tensor graph
    # priya_input_t = tf.placeholder(shape=[BATCH_SIZE, OUTPUT_SIZE - 1], dtype=tf.float32)
    # priya_label_t = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32)
    # priya_output_t = adversary_conv(OUTPUT_SIZE - 1)(priya_input_t)
    # # losses and optimizers
    # priya_loss_t = tf.losses.absolute_difference(priya_label_t, priya_output_t)
    # janice_loss_t = -tf.losses.absolute_difference(janice_true_t, priya_pred_t)
    # janice_optimizer = GEN_OPT.minimize(janice_loss_t)
    # priya_optimizer = OPP_OPT.minimize(priya_loss_t)
    #
    # # run TensorFlow session
    # with tf.train.SingularMonitoredSession() as sess:
    #     losses_janice = []
    #     losses_priya = []
    #     # train - the training loop is broken into sections that
    #     # are connected by fetches and feeds
    #     try:
    #         evaluate(sess, janice_output_t, janice_input_t, 0, 'janice')
    #
    #         for step in range(STEPS):
    #             batch_inputs = get_input_batch_np(BATCH_SIZE, MAX_VAL)
    #             # generate outputs
    #             janice_output_n = sess.run([janice_output_t],
    #                                        feed_dict={janice_input_t: batch_inputs})
    #             priya_input_n, priya_label_n = slice_gen_out(janice_output_n[0])
    #             # compute priya loss and update parameters
    #             priya_output_n = None
    #             priya_loss_epoch = None
    #             for adv in range(ADV_MULT):
    #                 _, priya_loss_epoch, priya_output_n = sess.run([priya_optimizer, priya_loss_t, priya_output_t],
    #                                                                feed_dict={priya_input_t: priya_input_n,
    #                                                                           priya_label_t: priya_label_n})
    #             # compute janice loss and update parameters
    #             _, janice_loss_epoch = sess.run([janice_optimizer, janice_loss_t],
    #                                             feed_dict={priya_pred_t: priya_output_n,
    #                                                        janice_input_t: batch_inputs})
    #
    #             # log and evaluate
    #             if step % LOG_EVERY_N == 0:
    #                 debug.print_step(step, janice_loss_epoch, priya_loss_epoch)
    #                 losses_janice.append(janice_loss_epoch)
    #                 losses_priya.append(priya_loss_epoch)
    #
    #     except KeyboardInterrupt:
    #         print('[INTERRUPTED BY USER] -- evaluating')
    #
    #     # produce output
    #     files.write_to_file(losses_janice, PLOT_DIR + '/janice_loss.txt')
    #     files.write_to_file(losses_priya, PLOT_DIR + '/priya_loss.txt')
    #     evaluate(sess, janice_output_t, janice_input_t, 1, 'janice')


# def generator(noise) -> tf.Tensor:
#     """ Symbolic tensor operations representing the generator neural network.
#
#         :param noise: generator input tensor with arbitrary distribution
#     """
#     inputs = tf.reshape(noise, [-1, 2])
#     outputs = fully_connected(inputs, GEN_WIDTH, activation=leaky_relu)
#     outputs = fully_connected(outputs, GEN_WIDTH, activation=leaky_relu)
#     outputs = fully_connected(outputs, GEN_WIDTH, activation=leaky_relu)
#     outputs = fully_connected(outputs, GEN_WIDTH, activation=leaky_relu)
#     outputs = fully_connected(outputs, OUTPUT_SIZE, activation=modulo(MAX_VAL))
#     return outputs


# def adversary_conv(size: int):
#     """ Returns a function representing the symbolic Tensor operations computed
#         by the convolutional opponent architecture, where the input layer of the
#         network is given as an argument.
#
#         :param size: the size of the adversary's input layer
#     """
#
#     def closure(inputs, unused_conditioning=None, weight_decay=2.5e-5, is_training=True) -> tf.Tensor:
#         input_layer = tf.reshape(inputs, [-1, size])
#         outputs = tf.expand_dims(input_layer, 2)
#         outputs = conv1d(outputs, filters=4, kernel_size=2, strides=1, padding='same', activation=leaky_relu)
#         outputs = conv1d(outputs, filters=4, kernel_size=2, strides=1, padding='same', activation=leaky_relu)
#         outputs = conv1d(outputs, filters=4, kernel_size=2, strides=1, padding='same', activation=leaky_relu)
#         outputs = conv1d(outputs, filters=4, kernel_size=2, strides=1, padding='same', activation=leaky_relu)
#         outputs = max_pooling1d(outputs, pool_size=2, strides=1)
#         outputs = flatten(outputs)
#         outputs = fully_connected(outputs, 4, activation=leaky_relu)
#         outputs = fully_connected(outputs, 1, activation=leaky_relu)
#         return outputs
#
#     return closure


# def evaluate(sess: tf.Session, gen_output, gen_input, iteration: int, name: str):
#     """ Evaluates the model by running it on the evaluation input and producing
#         a file of the outputs.
#
#         :param sess: running TensorFlow session
#         :param gen_output: tensor node in computational graph holding generator outputs
#         :param gen_input: tensor node in computational graph holding generator inputs
#         :param iteration: current training iteration, used for logging
#         :param name: name of the generator being evaluated
#     """
#     print('\n----------\n' + 'Running evaluation ...\n')
#     output = []
#
#     for batch in range(EVAL_BATCHES):
#         gen_out_vals = sess.run(gen_output, {gen_input: EVAL_DATA[batch]})
#         output.extend(gen_out_vals)
#     files.write_output_file(output, SEQN_DIR + str(iteration) + '_' + name + '.txt')
#     print('[DONE]\n' + '----------\n\n')


if __name__ == '__main__':
    main()
