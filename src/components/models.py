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
from components.operations import combine_generated_and_reference_tf, slice_gen_out_tf
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
        self.losses = {
            'generator': [],
            'discriminator': []
        }
        # other parameters
        self.input_width = input_width
        self.gen_out_width = gen_out_width
        self.max_val = max_val
        self.adv_multiplier = adv_multiplier

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
        for _ in range(steps):
            loss = {'disc': [], 'gen': []}
            # update discriminator
            for _ in range(self.adv_multiplier):
                gen_samples = self.generator(self.noise_prior(int(batch_size / 2), self.input_width, self.max_val))
                real_samples = self.reference_distribution(int(batch_size / 2), self.gen_out_width, self.max_val)
                disc_x, disc_y = combine_generated_and_reference_tf(gen_samples, real_samples)
                with tf.GradientTape() as tape:
                    loss['disc'] = self.loss_functions['discriminator'](disc_y, self.discriminator(disc_x))
                    disc_gradients = tape.gradient(loss['disc'], self.discriminator.variables)
                    self.optimizers['discriminator'].apply_gradients(
                        zip(disc_gradients, self.discriminator.variables))
            # update generator
            with tf.GradientTape() as tape:
                gen_samples = self.generator(self.noise_prior(int(batch_size), self.input_width, self.max_val))
                disc_out = self.discriminator(gen_samples)
                loss['gen'] = self.loss_functions['generator'](tf.zeros(shape=(batch_size,)), disc_out)
                gen_gradients = tape.gradient(loss['gen'], self.generator.variables)
                self.optimizers['generator'].apply_gradients(zip(gen_gradients, self.generator.variables))
            self.losses['discriminator'].append(loss['disc'])
            self.losses['generator'].append(loss['gen'])

    def predict(self, x, batch_size):
        return self.generator.predict(x, batch_size)

    def get_recorded_losses(self) -> dict:
        return self.losses


class PredGAN:
    """ A predictive GAN model, consisting of a generator and a predictor. """

    def __init__(self, input_width=2, gen_width=10, gen_out_width=8, max_val=65535, pred_width=10, adv_multiplier=3):
        # models
        self.generator = generator(input_width, gen_width, gen_out_width, max_val)
        self.predictor = adversary(gen_out_width - 1, 1, max_val)
        # data sources
        self.noise_prior = None
        # optimization
        self.optimizers = dict.fromkeys(['generator', 'predictor'])
        self.loss_functions = dict.fromkeys(['generator', 'predictor'])
        self.losses = {
            'generator': [],
            'predictor': []
        }
        # other parameters
        self.input_width = input_width
        self.gen_out_width = gen_out_width
        self.max_val = max_val
        self.adv_multiplier = adv_multiplier

    def with_distributions(self, noise_prior) -> 'PredGAN':
        self.noise_prior = noise_prior
        return self

    def with_optimizers(self, gen_optimizer: tf.train.Optimizer, pred_optimizer: tf.train.Optimizer) -> 'PredGAN':
        self.optimizers['generator'] = gen_optimizer
        self.optimizers['predictor'] = pred_optimizer
        return self

    def with_loss_functions(self, gen_loss_fn, pred_loss_fn) -> 'PredGAN':
        self.loss_functions['generator'] = gen_loss_fn
        self.loss_functions['predictor'] = pred_loss_fn
        return self

    def train(self, batch_size, steps):
        for _ in range(steps):
            loss = {'pred': [], 'gen': []}
            # update predictor
            for _ in range(self.adv_multiplier):
                gen_samples = self.generator(self.noise_prior(int(batch_size), self.input_width, self.max_val))
                pred_x, pred_y = slice_gen_out_tf(gen_samples)
                with tf.GradientTape() as tape:
                    loss['pred'] = self.loss_functions['predictor'](pred_y, self.predictor(pred_x))
                    pred_gradients = tape.gradient(loss['pred'], self.predictor.variables)
                    self.optimizers['predictor'].apply_gradients(zip(pred_gradients, self.predictor.variables))
            # update generator
            with tf.GradientTape() as tape:
                gen_samples = self.generator(self.noise_prior(int(batch_size), self.input_width, self.max_val))
                pred_x, pred_y = slice_gen_out_tf(gen_samples)
                disc_out = self.predictor(pred_x)
                loss['gen'] = self.loss_functions['generator'](pred_y, disc_out)  # todo
                gen_gradients = tape.gradient(loss['gen'], self.generator.variables)
                self.optimizers['generator'].apply_gradients(zip(gen_gradients, self.generator.variables))
            self.losses['predictor'].append(loss['pred'])
            self.losses['generator'].append(loss['gen'])

    def predict(self, x, batch_size):
        return self.generator.predict(x, batch_size)

    def get_recorded_losses(self) -> dict:
        return self.losses


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
        Reshape(target_shape=(input_width, 1)),
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
