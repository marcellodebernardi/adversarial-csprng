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
-nodisc         SKIP DISCRIMINATIVE GAN: does not train discriminative GAN
-nopred         SKIP PREDICTIVE GAN: does not train predictive GAN
-long           LONG TRAINING: trains for 1,000,000 epochs
-short          SHORT TRAINING: trains for 10,000 epochs
-highlr         HIGH LEARNING RATE: increases the learning rate from default
-lowlr          LOW LEARNING RATE: decreases the learning rate from default
"""

import sys
import tensorflow as tf
import tensorflow.contrib.gan as tfgan
from tensorflow.python.layers.core import fully_connected, flatten
from tensorflow.python.layers.pooling import max_pooling1d
from tensorflow.python.layers.convolutional import conv1d
from tensorflow.python.ops.nn import leaky_relu
from components.activations import modulo
from components.operations import slice_gen_out
from components.inputs import get_input_tensor, get_input_numpy, get_eval_input_numpy
from utils import files, debug

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

# optimizers
GEN_OPT = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.9999, beta2=0.9999)
OPP_OPT = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.9999, beta2=0.9999)

# training settings
TRAIN = ['-nodisc' not in sys.argv, '-nopred' not in sys.argv]
STEPS = 1000000 if '-long' in sys.argv else 10000 if '-short' in sys.argv else 200000 if HPC_TRAIN else 40
PRE_STEPS = 100 if HPC_TRAIN else 5
ADV_MULT = 3

# logging and evaluation
EVAL_BATCHES = 400 if HPC_TRAIN else 10
EVAL_DATA = get_eval_input_numpy(10, EVAL_BATCHES, BATCH_SIZE)
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
        print('# DISCGAN - batch size: ', BATCH_SIZE)
        run_discgan()
    # train predictive GAN
    if TRAIN[1]:
        print('# PREDGAN - batch size: ', BATCH_SIZE)
        run_predgan()

    # print used settings for convenience
    print('\n[COMPLETE] (DISCGAN: ' + str(TRAIN[0]) + ', PREDGAN: ' + str(TRAIN[1]) + ')')


def run_discgan():
    """ Constructs and trains the discriminative GAN consisting of
        Jerry and Diego.
    """
    # code follows the examples from
    # https://github.com/tensorflow/models/blob/master/research/gan/tutorial.ipynb

    # build the GAN model
    discgan = tfgan.gan_model(
        generator_fn=generator,
        discriminator_fn=adversary_conv(OUTPUT_SIZE),
        real_data=tf.random_uniform(shape=[BATCH_SIZE, OUTPUT_SIZE]),
        generator_inputs=get_input_tensor(BATCH_SIZE, MAX_VAL)
    )
    # Build the GAN loss
    discgan_loss = tfgan.gan_loss(
        discgan,
        generator_loss_fn=tfgan.losses.least_squares_generator_loss,
        discriminator_loss_fn=tfgan.losses.least_squares_discriminator_loss
    )
    # Create the train ops, which calculate gradients and apply updates to weights.
    train_ops = tfgan.gan_train_ops(
        discgan,
        discgan_loss,
        generator_optimizer=GEN_OPT,
        discriminator_optimizer=OPP_OPT
    )
    # start TensorFlow session
    with tf.train.SingularMonitoredSession() as sess:
        pretrain_steps_fn = tfgan.get_sequential_train_steps(tfgan.GANTrainSteps(0, PRE_STEPS))
        train_steps_fn = tfgan.get_sequential_train_steps(tfgan.GANTrainSteps(1, ADV_MULT))
        global_step = tf.train.get_or_create_global_step()

        # pretrain discriminator
        print('\n\nPretraining ... ', end="", flush=True)
        try:
            pretrain_steps_fn(sess, train_ops, global_step, train_step_kwargs={})
        except KeyboardInterrupt:
            pass
        print('[DONE]\n\n')

        # train both models
        losses_jerry = []
        losses_diego = []
        try:
            evaluate(sess, discgan.generated_data, discgan.generator_inputs, 0, 'jerry')

            for step in range(STEPS):
                train_steps_fn(sess, train_ops, global_step, train_step_kwargs={})

                # if performed right number of steps, log
                if step % LOG_EVERY_N == 0:
                    sess.run([])
                    gen_l = discgan_loss.generator_loss.eval(session=sess)
                    disc_l = discgan_loss.discriminator_loss.eval(session=sess)

                    debug.print_step(step, gen_l, disc_l)
                    losses_jerry.append(gen_l)
                    losses_diego.append(disc_l)

        except KeyboardInterrupt:
            print('[INTERRUPTED BY USER] -- evaluating')

        # produce output
        files.write_to_file(losses_jerry, PLOT_DIR + '/jerry_loss.txt')
        files.write_to_file(losses_diego, PLOT_DIR + '/diego_loss.txt')
        evaluate(sess, discgan.generated_data, discgan.generator_inputs, 1, 'jerry')


def run_predgan():
    """ Constructs, trains and evaluates the predictive GAN consisting
        of Janice and Priya.
    """
    # tensor graphs for janice and priya are separated so that optimizers
    # only work on intended nodes. Values are passed between janice and
    # priya at runtime using fetches and feeds.

    # Variables suffixed with _t are tensors and form part of the computational
    # graph. Variables suffixed with _n hold numpy arrays and are used at runtime
    # with fetch and feed operations to pass values between the generator and
    # the predictor

    # janice tensor graph
    janice_input_t = tf.placeholder(shape=[BATCH_SIZE, 2], dtype=tf.float32)
    janice_output_t = generator(janice_input_t)
    janice_true_t = tf.strided_slice(janice_output_t, [0, -0], [BATCH_SIZE, 1], [1, 1])
    priya_pred_t = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32)
    # priya tensor graph
    priya_input_t = tf.placeholder(shape=[BATCH_SIZE, OUTPUT_SIZE - 1], dtype=tf.float32)
    priya_label_t = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32)
    priya_output_t = adversary_conv(OUTPUT_SIZE - 1)(priya_input_t)
    # losses and optimizers
    priya_loss_t = tf.losses.absolute_difference(priya_label_t, priya_output_t)
    janice_loss_t = -tf.losses.absolute_difference(janice_true_t, priya_pred_t)
    janice_optimizer = GEN_OPT.minimize(janice_loss_t)
    priya_optimizer = OPP_OPT.minimize(priya_loss_t)

    # run TensorFlow session
    with tf.train.SingularMonitoredSession() as sess:
        losses_janice = []
        losses_priya = []
        # train - the training loop is broken into sections that
        # are connected by fetches and feeds
        try:
            evaluate(sess, janice_output_t, janice_input_t, 0, 'janice')

            for step in range(STEPS):
                batch_inputs = get_input_numpy(BATCH_SIZE, MAX_VAL)
                # generate outputs
                janice_output_n = sess.run([janice_output_t],
                                           feed_dict={janice_input_t: batch_inputs})
                priya_input_n, priya_label_n = slice_gen_out(janice_output_n[0])
                # compute priya loss and update parameters
                priya_output_n = None
                priya_loss_epoch = None
                for adv in range(ADV_MULT):
                    _, priya_loss_epoch, priya_output_n = sess.run([priya_optimizer, priya_loss_t, priya_output_t],
                                                                   feed_dict={priya_input_t: priya_input_n,
                                                                              priya_label_t: priya_label_n})
                # compute janice loss and update parameters
                _, janice_loss_epoch = sess.run([janice_optimizer, janice_loss_t],
                                                feed_dict={priya_pred_t: priya_output_n,
                                                           janice_input_t: batch_inputs})

                # log and evaluate
                if step % LOG_EVERY_N == 0:
                    debug.print_step(step, janice_loss_epoch, priya_loss_epoch)
                    losses_janice.append(janice_loss_epoch)
                    losses_priya.append(priya_loss_epoch)

        except KeyboardInterrupt:
            print('[INTERRUPTED BY USER] -- evaluating')

        # produce output
        files.write_to_file(losses_janice, PLOT_DIR + '/janice_loss.txt')
        files.write_to_file(losses_priya, PLOT_DIR + '/priya_loss.txt')
        evaluate(sess, janice_output_t, janice_input_t, 1, 'janice')


def generator(noise) -> tf.Tensor:
    """ Symbolic tensor operations representing the generator neural network.

        :param noise: generator input tensor with arbitrary distribution
    """
    inputs = tf.reshape(noise, [-1, 2])
    outputs = fully_connected(inputs, GEN_WIDTH, activation=leaky_relu)
    outputs = fully_connected(outputs, GEN_WIDTH, activation=leaky_relu)
    outputs = fully_connected(outputs, GEN_WIDTH, activation=leaky_relu)
    outputs = fully_connected(outputs, GEN_WIDTH, activation=leaky_relu)
    outputs = fully_connected(outputs, OUTPUT_SIZE, activation=modulo(MAX_VAL))
    return outputs


def adversary_conv(size: int):
    """ Returns a function representing the symbolic Tensor operations computed
        by the convolutional opponent architecture, where the input layer of the
        network is given as an argument.

        :param size: the size of the adversary's input layer
    """

    def closure(inputs, unused_conditioning=None, weight_decay=2.5e-5, is_training=True) -> tf.Tensor:
        input_layer = tf.reshape(inputs, [-1, size])
        outputs = tf.expand_dims(input_layer, 2)
        outputs = conv1d(outputs, filters=4, kernel_size=2, strides=1, padding='same', activation=leaky_relu)
        outputs = conv1d(outputs, filters=4, kernel_size=2, strides=1, padding='same', activation=leaky_relu)
        outputs = conv1d(outputs, filters=4, kernel_size=2, strides=1, padding='same', activation=leaky_relu)
        outputs = conv1d(outputs, filters=4, kernel_size=2, strides=1, padding='same', activation=leaky_relu)
        outputs = max_pooling1d(outputs, pool_size=2, strides=1)
        outputs = flatten(outputs)
        outputs = fully_connected(outputs, 4, activation=leaky_relu)
        outputs = fully_connected(outputs, 1, activation=leaky_relu)
        return outputs

    return closure


def evaluate(sess: tf.Session, gen_output, gen_input, iteration: int, name: str):
    """ Evaluates the model by running it on the evaluation input and producing
        a file of the outputs.

        :param sess: running TensorFlow session
        :param gen_output: tensor node in computational graph holding generator outputs
        :param gen_input: tensor node in computational graph holding generator inputs
        :param iteration: current training iteration, used for logging
        :param name: name of the generator being evaluated
    """
    print('\n----------\n' + 'Running evaluation ...\n')
    output = []

    for batch in range(EVAL_BATCHES):
        gen_out_vals = sess.run(gen_output, {gen_input: EVAL_DATA[batch]})
        output.extend(gen_out_vals)
    files.write_output_file(output, SEQN_DIR + str(iteration) + '_' + name + '.txt')
    print('[DONE]\n' + '----------\n\n')


if __name__ == '__main__':
    main()
