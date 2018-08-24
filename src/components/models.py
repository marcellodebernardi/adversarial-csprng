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
This module contains the actual TensorFlow implementations of the generative
adversarial networks involved in the paper.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Reshape
from components.operations import combine_generated_and_reference_tf
from components.activations import scaled_sigmoid
from tqdm import tqdm


class GAN:
    """ A standard discriminative GAN model, consisting of a generator and a
    discriminator. """

    def __init__(self, input_width=2, gen_width=10, gen_out_width=8, max_val=65535, disc_width=10, adv_multiplier=3):
        # models
        self.generator = generator(input_width, gen_width, gen_out_width, max_val)
        self.discriminator = adversary(None, 1, 1)
        # data sources
        self.noise_prior = None
        self.reference_distribution = None
        # optimization
        self.optimizers = dict.fromkeys(['generator', 'discriminator'])
        self.loss_functions = dict.fromkeys(['generator', 'discriminator'])
        # other parameters
        self.input_width = input_width
        self.gen_out_width = gen_out_width
        self.max_val = max_val
        self.adv_multiplier = adv_multiplier
        self.losses = {
            'generator': [],
            'discriminator': []
        }

    def with_distributions(self, noise_prior, reference_distribution) -> 'GAN':
        self.noise_prior = noise_prior
        self.reference_distribution = reference_distribution
        return self

    def with_optimizers(self, gen_optimizer: tf.train.Optimizer, disc_optimizer: tf.train.Optimizer) -> 'GAN':
        self.optimizers['generator'] = gen_optimizer
        self.optimizers['discriminator'] = disc_optimizer
        return self

    def with_loss_functions(self, gen_loss_fn, disc_loss_fn) -> 'GAN':
        self.loss_functions['generator'] = gen_loss_fn
        self.loss_functions['discriminator'] = disc_loss_fn
        return self

    def train(self, batch_size, steps):
        for step in tqdm(range(steps)):
            disc_loss = 0
            gen_loss = 0
            for _ in range(self.adv_multiplier):
                # sample from reference distribution and from generator
                real = self.reference_distribution(int(batch_size / 2), self.gen_out_width, self.max_val)
                gen_output = self.generator(self.noise_prior(int(batch_size / 2), self.input_width, self.max_val))
                # combine real sample and generator output
                disc_in, disc_labels = combine_generated_and_reference_tf(gen_output, real)
                # update discriminator
                with tf.GradientTape() as tape:
                    disc_out = self.discriminator(disc_in)
                    disc_loss = self.loss_functions['discriminator'](disc_labels, disc_out)
                    disc_gradients = tape.gradient(disc_loss, self.discriminator.variables)
                    self.optimizers['discriminator'].apply_gradients(
                        zip(disc_gradients, self.discriminator.variables))
            # update generator
            with tf.GradientTape() as tape:
                gen_output = self.generator(self.noise_prior(int(batch_size), self.input_width, self.max_val))
                disc_out = self.discriminator(gen_output)
                gen_loss = self.loss_functions['generator'](tf.zeros(shape=(batch_size,)), disc_out)
                gen_gradients = tape.gradient(gen_loss, self.generator.variables)
                self.optimizers['generator'].apply_gradients(zip(gen_gradients, self.generator.variables))
            self.losses['discriminator'].append(disc_loss)
            self.losses['generator'].append(gen_loss)

    def predict(self, x, batch_size):
        return self.generator.predict(x, batch_size)

    def get_recorded_losses(self) -> dict:
        return self.losses


class PredGAN:
    """ A predictive GAN model, consisting of a generator and a predictor. """

    def __init__(self, adv_mult=3):
        self.generator = generator()
        self.predictor = adversary()
        self.adv_mult = adv_mult

    def compile_generator(self, optimizer, loss, metrics) -> 'PredGAN':
        self.generator.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self

    def compile_predictor(self, optimizer, loss, metrics) -> 'PredGAN':
        self.predictor.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self

    def train(self, x, batch_size, epochs=1):
        """ Train the PredGAN for specified number of epochs using the provided input
        data. Computation is done in batches of the specified size. """
        x = tf.reshape(x, [-1, batch_size, tf.shape(x)[1]])

        for epoch in range(epochs):
            for batch in range(tf.shape(x)[0]):
                pass

    def predict(self, x, batch_size) -> tf.Tensor:
        """ Return generator outputs for all inputs x. """
        return self.generator.predict(x, batch_size)


def generator(input_width, hidden_width, output_width, max_val) -> tf.keras.Model:
    """ Returns a tensor function representing the generator model. The resulting
    function takes the model input as an argument and returns the model's output.
    :param input_width: the size of the input layer
    :param hidden_width: the size of the hidden layers, if applicable
    :param output_width: the size of the output layer
    :param max_val: the maximum value allowable for output by the network
    :return:
    """
    model = tf.keras.Sequential([
        Dense(input_width, scaled_sigmoid(max_val)),
        Dense(hidden_width, scaled_sigmoid(max_val)),
        Dense(hidden_width, scaled_sigmoid(max_val)),
        Dense(hidden_width, scaled_sigmoid(max_val)),
        Dense(output_width, scaled_sigmoid(max_val))
    ])
    return model


def adversary(input_width, output_width, max_val) -> tf.keras.Model:
    """ Returns a tensor function representing the adversary model. The resulting
    function takes the model input as an argument and returns the model's output.
    :param input_width: the size of the input layer
    :param output_width: the size of the output layer
    :param max_val: the maximum value that a neuron can produce
    :return:
    """
    model = tf.keras.Sequential([
        Reshape(target_shape=(8, 1)),
        Conv1D(4, 2, 1, 'same', activation=scaled_sigmoid(max_val), name='conv_1'),
        Conv1D(4, 2, 1, 'same', activation=scaled_sigmoid(max_val), name='conv_2'),
        Conv1D(4, 2, 1, 'same', activation=scaled_sigmoid(max_val), name='conv_3'),
        Conv1D(4, 2, 1, 'same', activation=scaled_sigmoid(max_val), name='conv_4'),
        MaxPool1D(2, 1, name='maxpool_1'),
        Flatten(name='flatten_1'),
        Dense(4, activation=scaled_sigmoid(max_val), name='dense_1'),
        Dense(output_width, scaled_sigmoid(max_val), name='dense_2')
    ])
    return model
