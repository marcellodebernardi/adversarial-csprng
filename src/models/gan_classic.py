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
import numpy as np
from keras import Model
from keras.layers import Input
from models.gan import Gan
from data_sources import randomness_sources as rng
from utils import utils
from tqdm import tqdm


class ClassicGan(Gan):
    """Implementation of a classic generative adversarial network
    to the statistical randomness problem (i.e. 'approach 1').
    """

    def __init__(self, generator_spec, discriminator_spec, adversarial_spec, settings, io_params, train_params):
        """Constructs the classical GAN according to the specifications
        provided. Each specification variable is a dictionary."""
        # initialize common variables and generator
        Gan.__init__(self, generator_spec, discriminator_spec, adversarial_spec, settings, io_params, train_params)
        # discriminator attributes
        self.disc_types = discriminator_spec['types']
        self.disc_activations = discriminator_spec['activations']
        self.disc_loss = discriminator_spec['loss']
        self.disc_optimizer = discriminator_spec['optimizer']
        self.discriminator = self.create_discriminator()
        # connect GAN
        self.adversarial_optimizer = adversarial_spec['optimizer']
        self.adversarial_loss = adversarial_spec['loss']
        self.adversarial = self.connect_gan()

    def pretrain_discriminator(self, epochs):
        """Pre-trains the discriminator for a given number of epochs."""
        # record initial weights
        self.metrics.generator_weights_initial() \
            .extend(utils.flatten_irregular_nested_iterable(self.generator.get_weights()))
        self.metrics.predictor_weights_initial() \
            .extend(utils.flatten_irregular_nested_iterable(self.discriminator.get_weights()))
        # fit discriminator using sample
        x, y = self.construct_discriminator_sample()
        print(self.discriminator.fit(x, y, epochs, self.batch_size))

    def train(self, epochs, disc_mult):
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
            self.set_trainable(self.discriminator)
            for iteration in range(disc_mult):
                d_loss = self.discriminator.train_on_batch(x_disc, y_disc)
            # train generator
            self.set_trainable(self.discriminator, False)
            g_loss = self.adversarial.train_on_batch(x_gen, y_gen)
            # update loss metrics
            self.metrics.predictor_loss().append(d_loss)
            self.metrics.generator_loss().append(g_loss)
        # update final node weight metrics
        self.metrics.generator_weights_final().extend(
            utils.flatten_irregular_nested_iterable(self.generator.get_weights()))
        self.metrics.predictor_weights_final().extend(
            utils.flatten_irregular_nested_iterable(self.discriminator.get_weights()))
        return self.metrics

    def get_generator(self) -> Model:
        """Returns a reference to the internal generator Keras model."""
        return self.generator

    def get_discriminator(self) -> Model:
        """Returns a reference to the internal discriminator Keras model."""
        return self.discriminator

    def get_adversarial(self) -> Model:
        """Returns a reference to the internal adversarial Keras model."""
        return self.adversarial

    def create_discriminator(self) -> Model:
        """Initializes and compiles the discriminator model. Returns a reference to
        the model.
        """
        # inputs and first operation
        inputs_disc = Input(shape=(self.out_seq_len,))
        operations_disc = self.layer(self.disc_types[0],
                                     self.out_seq_len if len(self.gen_types) > 1 else 1,
                                     self.disc_activations[0])(inputs_disc)
        # additional operations if depth > 1
        for layer_index in range(1, len(self.gen_types)):
            operations_disc = self.layer(self.disc_types[layer_index],
                                         self.out_seq_len if layer_index < len(self.disc_types) - 1 else 1,
                                         self.disc_activations[layer_index])(operations_disc)
        # compile and return model
        discriminator = Model(inputs_disc, operations_disc, name='discriminator')
        discriminator.compile(self.disc_optimizer, self.disc_loss)
        return discriminator

    def connect_gan(self):
        """Performs the connection of the generator and discriminator into
        a GAN, returning a reference to the adversarial model.
        """
        operations_adv = self.generator(self.generator_input)
        operations_adv = self.discriminator(operations_adv)
        # compile and return
        adversarial = Model(self.generator_input, operations_adv, name='adversarial')
        adversarial.compile(self.adversarial_optimizer, self.adversarial_loss)
        return adversarial

    def construct_discriminator_sample(self) -> (np.ndarray, np.ndarray):
        """Constructs a sample batch which includes both truly random
        sequences and sequences produced by the generator, with the
        associated labels"""
        # todo get rid of magic numbers
        # generate discriminator inputs
        true = rng.get_random_sequence(self.max_val, self.out_seq_len, int(self.dataset_size / 2))
        generated = self.generator.predict(rng.get_seed_dataset(self.max_val, self.seed_len, 1, int(self.dataset_size / 2)))
        x = np.concatenate((true, generated))
        # add correct labels and return
        true_labels = np.zeros((int(self.dataset_size / 2),))
        false_labels = np.ones((int(self.dataset_size / 2),))
        y = np.concatenate((true_labels, false_labels))
        return x, y

    def construct_generator_sample(self) -> (np.ndarray, np.ndarray):
        """Constructs a sample for training the generator."""
        x = rng.get_seed_dataset(self.max_val, self.seed_len, 1, self.dataset_size)
        y = np.zeros((self.dataset_size,))
        return x, y
