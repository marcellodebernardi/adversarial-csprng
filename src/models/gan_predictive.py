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
from models.gan import Gan
from models.operations import drop_last_value
from models.metrics import Metrics
from keras import Model
from keras.layers import Input, Lambda
from tqdm import tqdm
import utils.utils as utils
import numpy as np


class PredictiveGan(Gan):
    def __init__(self, generator_spec, predictor_spec, adversarial_spec, settings, io_params, train_params):
        # common attributes
        Gan.__init__(self, generator_spec, predictor_spec, adversarial_spec, settings, io_params, train_params)
        # predictor attributes
        self.pred_types = predictor_spec['types']
        self.pred_activations = predictor_spec['activations']
        self.pred_loss = predictor_spec['loss']
        self.pred_optimizer = predictor_spec['optimizer']
        self.predictor = self.__create_predictor()
        # connect GAN
        self.adversarial_optimizer = adversarial_spec['optimizer']
        self.adversarial_loss = adversarial_spec['loss']
        self.adversarial = self.__connect_gan()

    def get_generator(self) -> Model:
        """Returns a reference to the internal generator Keras model."""
        return self.generator

    def get_predictor(self) -> Model:
        """Returns a reference to the internal predictor Keras model."""
        return self.predictor

    def get_adversarial(self) -> Model:
        """Returns a reference to the internal adversarial Keras model."""
        return self.adversarial

    def pretrain_predictor(self, seed_dataset, epochs):
        """Pre-trains the predictor network on its own. Used before the
        adversarial training of both models is started."""
        for epoch in range(epochs):
            epoch_pretrain_loss = []
            for generator_input in seed_dataset:
                generator_output = self.generator.predict_on_batch(generator_input)
                predictor_input, predictor_output = utils.split_generator_output(generator_output, 1)
                epoch_pretrain_loss.append(self.predictor.train_on_batch(predictor_input, predictor_output))
            self.metrics.predictor_pretrain_loss().append(np.mean(epoch_pretrain_loss))

    def train(self, seed_dataset, epochs, pred_mult):
        """Trains the adversarial model on the given dataset of seed values, for the
        specified number of epochs. The seed dataset must be 3-dimensional, of the form
        [batch, seed, seed_component]. Each 'batch' in the dataset can be of any size,
        including 1, allowing for online training, batch training, and mini-batch training.
        """
        if len(np.shape(seed_dataset)) != 3:
            raise ValueError('Seed dataset has ' + str(len(np.shape(seed_dataset))) + ' dimension(s), should have 3')

        self.metrics.generator_weights_initial().extend(utils.flatten_irregular_nested_iterable(self.generator.get_weights()))
        self.metrics.predictor_weights_initial().extend(utils.flatten_irregular_nested_iterable(self.predictor.get_weights()))

        # each epoch train on entire dataset
        for epoch in tqdm(range(epochs), desc='Train: '):
            epoch_gen_losses = []
            epoch_pred_losses = []
            # the length of generator input determines whether training
            # is effectively batch training, mini-batch training or
            # online training. This is a property of the dataset
            # todo should not be a property of the dataset
            # todo split into separate procedures
            for generator_input in seed_dataset:
                generator_output = self.generator.predict_on_batch(generator_input)
                self.metrics.generator_outputs().extend(generator_output.flatten())
                self.metrics.generator_avg_outputs().append(np.mean(generator_output.flatten()))

                predictor_input, predictor_output = utils.split_generator_output(generator_output, 1)

                # train predictor
                self.set_trainable(self.predictor)
                for i in range(pred_mult):
                    epoch_pred_losses.append(self.predictor.train_on_batch(predictor_input, predictor_output))

                # train generator
                self.set_trainable(self.predictor, False)
                epoch_gen_losses.append(self.adversarial.train_on_batch(generator_input, predictor_output))

            self.metrics.generator_loss().append(np.mean(epoch_gen_losses))
            self.metrics.predictor_loss().append(np.mean(epoch_pred_losses))

        self.metrics.generator_weights_final().extend(utils.flatten_irregular_nested_iterable(self.generator.get_weights()))
        self.metrics.predictor_weights_final().extend(utils.flatten_irregular_nested_iterable(self.predictor.get_weights()))
        return self.metrics

    def __create_predictor(self):
        """Returns a compiled predictor model"""
        # inputs
        inputs_pred = Input(shape=(self.out_seq_len - 1,))
        operations_pred = self.layer(self.pred_types[0],
                                     self.out_seq_len if len(self.pred_types) > 1 else 1,
                                     self.pred_activations[0])(inputs_pred)
        # additional layers if depth > 1
        for layer_index in range(1, len(self.pred_types)):
            operations_pred = self.layer(self.pred_types[layer_index],
                                         self.out_seq_len if layer_index != len(self.pred_types) - 1 else 1,
                                         self.pred_activations[layer_index])(operations_pred)
        # compile and return model
        predictor = Model(inputs_pred, operations_pred, name='predictor')
        predictor.compile(self.pred_optimizer, self.pred_loss)
        return predictor

    def __connect_gan(self):
        """Performs the connection of the generator and predictor into
        a GAN, returning a reference to the adversarial model.
        """
        # adversarial connection
        operations_adv = self.generator(self.generator_input)
        operations_adv = Lambda(
            drop_last_value(self.out_seq_len, self.batch_size),
            name='adversarial_drop_last_value')(operations_adv)
        operations_adv = self.predictor(operations_adv)
        # compile and return
        adversarial = Model(self.generator_input, operations_adv, name='adversarial')
        adversarial.compile(self.adversarial_optimizer, self.adversarial_loss)
        return adversarial
