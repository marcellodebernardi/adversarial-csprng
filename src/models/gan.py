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
"""
This module contains a base class defining a generative adversarial
network specifically as used in this project. That is, this module
does not provide general functionality that could be transferred to
another context, but rather a wrapper for functionality specific
to this project, in order to decrease cognitive load.
"""

from models.metrics import Metrics
from keras import Model
from keras.layers import Input, Dense, SimpleRNN, GRU, LSTM, Conv1D, Conv2D, ConvLSTM2D, Dropout
from utils.utils import eprint
from tqdm import tqdm


class Gan:
    """ Base class for the generative adversarial networks used in this
    project.
    """
    def __init__(self, generator_spec, adversary_spec, adversarial_spec, settings, io_params, train_params):
        # check for back network definitions
        self.__check_parameters([generator_spec, adversary_spec, adversarial_spec])
        # attributes common to all GANs
        self.name = adversarial_spec['name']
        self.seed_len = generator_spec['seed_len']
        self.out_seq_len = generator_spec['out_seq_len']
        self.max_val = io_params['max_val']
        self.metrics = Metrics()
        # training parameters
        self.batch_mode = settings['batch_mode']
        self.batch_size = settings['batch_size']
        self.pretrain_epochs = train_params['pretrain_epochs']
        self.train_epochs = train_params['epochs']
        self.dataset_size = settings['dataset_size']
        # generator attributes
        self.gen_types = generator_spec['types']
        self.gen_depth = generator_spec['depth']
        self.gen_activation = generator_spec['activation']
        self.gen_loss = generator_spec['loss']
        self.gen_optimizer = generator_spec['optimizer']
        self.generator_input, self.generator = self.__create_generator()

    def evaluate(self, seed_dataset):
        # todo should two approaches have separate evaluations?
        """Performs evaluation of the generator/adversarial model and updates
        the metrics object with the evaluation results.
        """
        for generator_input in tqdm(seed_dataset, desc='Eval: '):
            generator_output = self.generator.predict_on_batch(generator_input)
            self.metrics.generator_eval_outputs().extend(generator_output.flatten())

    def get_name(self):
        """Returns the name of the network."""
        return self.name

    def get_metrics(self):
        """Returns metrics collected for the model."""
        return self.metrics

    def __create_generator(self):
        """Initializes and compiles the generator model. Returns a reference to
        the input Tensor and the generator model."""
        # inputs and first layer
        inputs_gen = Input(shape=(self.seed_len,))
        operations_gen = self.__layer(self.gen_types[0], self.out_seq_len, self.gen_activation)(inputs_gen)
        # more layers if depth > 1
        for layer_index in range(1, self.gen_depth):
            type_index = layer_index if layer_index < len(self.gen_types) else len(self.gen_types) - 1
            operations_gen = self.__layer(self.gen_types[type_index], self.out_seq_len, self.gen_activation)(operations_gen)
        # compile and return
        generator = Model(inputs_gen, operations_gen, name='generator')
        generator.compile(self.gen_optimizer, self.gen_loss)
        return inputs_gen, generator

    @staticmethod
    def __layer(l_type, units, activation):
        """From a selection of possible layer types, uses l_type to pick a
        layer to return. The layer is defined by the given parameters."""
        if l_type == 'dense':
            return Dense(units, activation=activation)
        elif l_type == 'simple_rnn':
            return SimpleRNN(units, activation=activation)
        elif l_type == 'gru':
            return GRU(units, activation=activation)
        elif l_type == 'lstm':
            return LSTM(units, activation=activation)
        elif l_type == 'conv1d':
            return Conv1D(units, activation=activation)
        elif l_type == 'conv2d':
            return Conv2D(units, activation=activation)
        elif l_type == 'conv_lstm2d':
            return ConvLSTM2D(units, activation=activation)
        elif l_type == 'dropout':
            return Dropout(units, activation=activation)
        else:
            raise ValueError('Unrecognized layer type ' + str(l_type))

    @staticmethod
    def __check_parameters(specifications: list):
        """Checks whether the network definition parameters are correct, and prints
        warnings for parameters that are usable, but likely to be incorrect."""
        for specification in specifications:
            if specification['depth'] < 1:
                raise ValueError('Model depth must be at least 1')
            if len(specification['types']) < 1:
                raise ValueError('Types list is empty, must contain at least one type')
            elif len(specification['types']) > specification['depth']:
                eprint('Warning: ' + specification['name'] + ' number of layer types exceeds network depth.')
            elif len(specification['types']) < specification['depth']:
                eprint('Warning: ' + specification['name'] + ' number of layer types is less than network depth.')

    @staticmethod
    def __set_trainable(model: Model, trainable: bool = True):
        """Helper method that sets the trainability of all of a model's
        parameters."""
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable
