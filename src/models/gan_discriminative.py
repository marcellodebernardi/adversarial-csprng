# Marcello De Bernardi, Queen Mary University of London
#
# An exploratory proof-of-concept implementation of a CSPRNG
# (cryptographically secure pseudo-random number generator) using
# adversarially trained neural networks. The work was inspired by
# the findings of Abadi & Andersen, outlined in the paper
# "Learning to Protect Communications with Adversarial
# Networks", available at https://arxiv.org/abs/1610.06918.
#
# The original implementation by Abadi is available at
# https://github.com/tensorflow/models/tree/master/research/adversarial_crypto
# =================================================================
import random
import numpy as np
from keras import Model
from keras.layers import Input, Dense, SimpleRNN, Reshape, Flatten, Conv1D
from keras.activations import linear, softmax
from keras.optimizers import adagrad
from models.activations import modulo, absolute
from models.losses import loss_adv, loss_discriminator
from models.metrics import Metrics
from data_sources import randomness_sources as data
from utils import utils
from utils import vis_utils
from tqdm import tqdm


class DiscriminativeGan:
    """Implementation of a classic generative adversarial network
    to the statistical randomness problem (i.e. 'approach 1').
    """
    def __init__(self, dataset_size=100, max_val=100, seed_length=10, output_length=300, lr=0.1, cv=1):
        """Constructs the classical GAN according to the specifications
        provided. Each specification variable is a dictionary."""
        self.metrics = Metrics()
        self.dataset_size = dataset_size
        self.max_val = max_val
        self.seed_length = seed_length
        self.output_length = output_length
        # generator
        generator_inputs = Input(shape=(seed_length,))
        generator_outputs = Dense(output_length, activation=linear)(generator_inputs)
        generator_outputs = Reshape(target_shape=(5, 60))(generator_outputs)
        generator_outputs = SimpleRNN(60, return_sequences=True, activation=linear)(generator_outputs)
        generator_outputs = Flatten()(generator_outputs)
        generator_outputs = Dense(output_length, activation=modulo(max_val))(generator_outputs)
        self.generator = Model(generator_inputs, generator_outputs)
        self.generator.compile('adagrad', 'binary_crossentropy')
        vis_utils.plot_network_graphs(self.generator, 'disc_generator')
        # discriminator
        discriminator_inputs = Input(shape=(output_length,))
        discriminator_outputs = Dense(output_length, activation=linear)(discriminator_inputs)
        discriminator_outputs = Reshape(target_shape=(5, 60))(discriminator_outputs)
        discriminator_outputs = Conv1D(int(output_length / 4), 4)(discriminator_outputs)
        discriminator_outputs = Flatten()(discriminator_outputs)
        discriminator_outputs = Dense(int(output_length / 3), activation=linear)(discriminator_outputs)
        discriminator_outputs = Dense(1, activation=softmax)(discriminator_outputs)
        self.discriminator = Model(discriminator_inputs, discriminator_outputs)
        self.discriminator.compile(adagrad(lr=lr, clipvalue=cv), loss_discriminator)
        vis_utils.plot_network_graphs(self.discriminator, 'disc_adversary')
        # adversarial model
        operations_adv = self.generator(generator_inputs)
        operations_adv = self.discriminator(operations_adv)
        self.adversarial = Model(generator_inputs, operations_adv)
        self.adversarial.compile('adagrad', loss_adv(loss_discriminator))
        vis_utils.plot_network_graphs(self.adversarial, 'disc_gan')

    def pretrain_discriminator(self, epochs, batch_size):
        """Pre-trains the discriminator for a given number of epochs."""
        # record initial weights
        self.metrics.generator_weights_initial() \
            .extend(utils.flatten_irregular_nested_iterable(self.generator.get_weights()))
        self.metrics.predictor_weights_initial() \
            .extend(utils.flatten_irregular_nested_iterable(self.discriminator.get_weights()))
        # fit discriminator using sample
        x, y = self.construct_discriminator_sample()
        # todo unstable
        self.discriminator.fit(x, y, batch_size, epochs, verbose=0)

    def train(self, batch_size, epochs, disc_mult):
        """Trains the adversarial model on the given dataset of seed values, for the
        specified number of epochs. The seed dataset must be 3-dimensional, of the form
        [batch, seed, seed_component]. Each 'batch' in the dataset can be of any size,
        including 1, allowing for online training, batch training, and mini-batch training.
        """
        x_disc, y_disc = self.construct_discriminator_sample()
        x_gen, y_gen = self.construct_generator_sample()
        # each epoch train on entire dataset
        for epoch in tqdm(range(epochs), desc='Train: '):
            # train discriminator
            d_loss = 0
            utils.set_trainable(self.discriminator)
            for iteration in range(disc_mult):
                d_loss = self.discriminator.fit(x_disc, y_disc, batch_size, verbose=0)
            # train generator
            utils.set_trainable(self.discriminator, False)
            g_loss = self.adversarial.fit(x_gen, y_gen, batch_size, verbose=0)
            # update loss metrics
            self.metrics.predictor_loss().append(d_loss)
            self.metrics.generator_loss().append(g_loss)
        # update final node weight metrics
        self.metrics.generator_weights_final().extend(
            utils.flatten_irregular_nested_iterable(self.generator.get_weights()))
        self.metrics.predictor_weights_final().extend(
            utils.flatten_irregular_nested_iterable(self.discriminator.get_weights()))
        return self.metrics

    def get_model(self) -> (Model, Model, Model):
        """Returns the three underlying Keras models."""
        return self.generator, self.discriminator, self.adversarial

    def construct_discriminator_sample(self) -> (np.ndarray, np.ndarray):
        """Constructs a sample batch which includes both truly random
        sequences and sequences produced by the generator, with the
        associated labels"""
        # todo get rid of magic numbers
        # generate discriminator inputs
        true = data.get_random_sequence(self.max_val, self.output_length, int(self.dataset_size / 2))
        generated = self.generator.predict(
            data.get_seed_dataset(self.max_val, self.seed_length, 1, int(self.dataset_size / 2)))
        print(np.shape(true))
        print(np.shape(generated))
        x = np.concatenate((true, generated))
        # add correct labels and return
        true_labels = np.zeros((int(self.dataset_size / 2),))
        false_labels = np.ones((int(self.dataset_size / 2),))
        y = np.concatenate((true_labels, false_labels))
        return x, y

    def construct_generator_sample(self) -> (np.ndarray, np.ndarray):
        """Constructs a sample for training the generator."""
        x = data.get_seed_dataset(self.max_val, self.seed_length, 1, self.dataset_size)
        y = np.zeros((self.dataset_size,))
        return x, y

