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
"""

import sys
import tensorflow as tf
import tensorflow.contrib.gan as tfgan
from tensorflow.python.layers.core import fully_connected, flatten
from tensorflow.python.layers.pooling import max_pooling1d
from tensorflow.python.layers.convolutional import conv1d
from tensorflow.python.ops.nn import leaky_relu
from models.activations import modulo
from utils import utils, input, operations, debug

# main settings
HPC_TRAIN = '-t' not in sys.argv  # set to true when training on HPC to collect data

# hyper-parameters
OUTPUT_SIZE = 8
MAX_VAL = 15
OUTPUT_BITS = 4
BATCH_SIZE = 2046 if HPC_TRAIN else 10
LEARNING_RATE = 0.008
CLIP_VALUE = 0.03
GEN_WIDTH = 30 if HPC_TRAIN else 10
DATA_TYPE = tf.float64

# optimizers
GEN_OPT = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.9999, beta2=0.9999)
OPP_OPT = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.9999, beta2=0.9999)

# training settings
TRAIN = ['-nodisc' not in sys.argv, '-nopred' not in sys.argv]
EVAL_EVERY_N = 15000 if HPC_TRAIN else 10
STEPS = 150000 if HPC_TRAIN else 40
PRE_STEPS = 100 if HPC_TRAIN else 5
ADV_MULT = 3
SEND_REPORT = '-email' in sys.argv

# logging and evaluation
EVAL_BATCHES = int(50000 / BATCH_SIZE) if HPC_TRAIN else 10
EVAL_DATA = input.get_eval_input_numpy(10, EVAL_BATCHES, BATCH_SIZE, MAX_VAL)
LOG_EVERY_N = 10 if HPC_TRAIN else 1
PLOT_DIR = '../output/plots/'
DATA_DIR = '../output/data/'
SEQN_DIR = '../output/sequences/'
GRAPH_DIR = '../output/model_graphs/'


def main():
    """ Constructs the neural networks, trains them, and logs
    all relevant information."""
    # train discriminative GAN
    if TRAIN[0]:
        print('# DISCGAN - batch size: ', BATCH_SIZE)
        run_discgan()
    # train predictive GAN
    if TRAIN[1]:
        print('# PREDGAN - batch size: ', BATCH_SIZE)
        run_predgan()

    # print settings for convenience
    print('\n[COMPLETE] (DISCGAN: ' + str(TRAIN[0]) + ', PREDGAN: ' + str(TRAIN[1]) + ')')


def run_discgan():
    """ Constructs and trains the discriminative GAN consisting of
    Jerry and Diego. """
    # build the GAN model
    discgan = tfgan.gan_model(
        generator_fn=generator,
        discriminator_fn=adversary_conv(OUTPUT_SIZE),
        real_data=tf.random_uniform(shape=[BATCH_SIZE, OUTPUT_SIZE]),
        generator_inputs=input.get_input_tensor(BATCH_SIZE, MAX_VAL)
    )

    # Build the GAN loss.
    discgan_loss = tfgan.gan_loss(
        discgan,
        generator_loss_fn=tfgan.losses.least_squares_generator_loss,
        discriminator_loss_fn=tfgan.losses.least_squares_discriminator_loss)

    # Create the train ops, which calculate gradients and apply updates to weights.
    train_ops = tfgan.gan_train_ops(
        discgan,
        discgan_loss,
        generator_optimizer=GEN_OPT,
        discriminator_optimizer=OPP_OPT)

    # run session
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

        # train
        try:
            evaluation_counter = 0
            print('Training ...')
            for step in range(STEPS):
                train_steps_fn(sess, train_ops, global_step, train_step_kwargs={})
                if step % LOG_EVERY_N == 0:
                    sess.run([])
                    debug.print_step(step, discgan_loss.generator_loss.eval(session=sess),
                                     discgan_loss.discriminator_loss.eval(session=sess))
                if step % EVAL_EVERY_N == 0:
                    evaluate(sess, discgan.generated_data, discgan.generator_inputs, evaluation_counter, 'jerry')
                    evaluation_counter += 1

        except KeyboardInterrupt:
            print('[INTERRUPTED BY USER] -- evaluating')

        # produce output
        evaluate(sess, discgan.generated_data, discgan.generator_inputs, -1, 'jerry')


def run_predgan():
    """ Constructs, trains and evaluates the predictive GAN consisting
    of Janice and Priya. """
    # tensor graphs for janice and priya are separated so that optimizers
    # only work on intended nodes. Values are passed between janice and
    # priya at runtime using fetches and feeds.

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
    priya_loss = tf.losses.mean_squared_error(priya_label_t, priya_output_t)
    janice_loss = -tf.losses.mean_squared_error(janice_true_t, priya_pred_t)
    janice_optimizer = GEN_OPT.minimize(janice_loss)
    priya_optimizer = OPP_OPT.minimize(priya_loss)

    # run session
    with tf.train.SingularMonitoredSession() as sess:
        # train
        # the training loop is broken into sections that are connected by
        # fetches and feeds
        try:
            evaluation_counter = 0

            for step in range(STEPS):
                batch_inputs = input.get_input_numpy(BATCH_SIZE, MAX_VAL)
                # generate
                janice_output_n = sess.run([janice_output_t],
                                           feed_dict={janice_input_t: batch_inputs})
                priya_input_n, priya_label_n = operations.slice_gen_out(janice_output_n[0])
                # update priya
                priya_output_n = None
                priya_loss_epoch = None
                for adv in range(ADV_MULT):
                    _, priya_loss_epoch, priya_output_n = sess.run([priya_optimizer, priya_loss, priya_output_t],
                                                                   feed_dict={priya_input_t: priya_input_n,
                                                                              priya_label_t: priya_label_n})
                # update janice
                _, janice_loss_epoch = sess.run([janice_optimizer, janice_loss],
                                                feed_dict={priya_pred_t: priya_output_n,
                                                           janice_input_t: batch_inputs})

                if step % LOG_EVERY_N == 0:
                    debug.print_step(step, janice_loss_epoch, priya_loss_epoch)
                if step % EVAL_EVERY_N == 0:
                    evaluate(sess, janice_output_t, janice_input_t, evaluation_counter, 'janice')
                    evaluation_counter += 1

        except KeyboardInterrupt:
            print('[INTERRUPTED BY USER] -- evaluating')

        # produce output
        evaluate(sess, janice_output_t, janice_input_t, -1, 'janice')


def generator(noise) -> tf.Tensor:
    """ Symbolic tensor operations representing the generator neural network. """
    input_layer = tf.reshape(noise, [-1, 2])
    outputs = fully_connected(input_layer, GEN_WIDTH, activation=leaky_relu)
    outputs = fully_connected(outputs, GEN_WIDTH, activation=leaky_relu)
    outputs = fully_connected(outputs, GEN_WIDTH, activation=leaky_relu)
    outputs = fully_connected(outputs, GEN_WIDTH, activation=leaky_relu)
    outputs = fully_connected(outputs, OUTPUT_SIZE, activation=modulo(MAX_VAL))
    return outputs


def adversary_conv(size: int):
    """ Returns a function representing the symbolic Tensor operations computed
        by the convolutional opponent architecture, where the input layer of the
        network is given as an argument. """

    def closure(inputs, unused_conditioning=None, weight_decay=2.5e-5, is_training=True):
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
    output = []
    for batch in range(EVAL_BATCHES):
        j_out = sess.run(gen_output, {gen_input: EVAL_DATA[batch]})
        output.extend(j_out)
    utils.generate_output_file(output, OUTPUT_BITS, SEQN_DIR + str(iteration) + '_' + name + '.txt')
    utils.log_to_file(output, SEQN_DIR + str(iteration) + '_' + name + '_eval_sequence.txt')


if __name__ == '__main__':
    main()
