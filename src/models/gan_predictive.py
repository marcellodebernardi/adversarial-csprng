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
from models.operations import drop_last_value
from models.metrics import Metrics
from models.losses import loss_predictor, loss_adv
from models.activations import modulo
from keras import Model
from keras.layers import Input, Lambda, Dense, SimpleRNN, LSTM
from keras.activations import linear, relu
from tqdm import tqdm
import utils.utils as utils
import numpy as np


class PredictiveGan:
    def __init__(self, max_val=100, timesteps=5, seed_length=10, output_length=30, lr=0.1, cv=1, batch_size=1):
        self.metrics = Metrics()
        # generator
        generator_inputs = Input(shape=(timesteps, seed_length))
        generator_outputs = Dense(output_length, activation=linear)(generator_inputs)
        generator_outputs = SimpleRNN(output_length, activation=linear)(generator_outputs)
        generator_outputs = Dense(output_length, activation=modulo(max_val))(generator_outputs)
        self.generator = Model(generator_inputs, generator_outputs)
        self.generator.compile('adagrad', 'binary_crossentropy')
        self.generator.summary()
        # predictor
        predictor_inputs = Input(shape=(timesteps, output_length - 1))
        predictor_outputs = LSTM(output_length - 1, activation=linear)(predictor_inputs)
        predictor_outputs = Dense(output_length, activation=linear)(predictor_outputs)
        predictor_outputs = Dense(1, activation=relu)(predictor_outputs)
        self.predictor = Model(predictor_inputs, predictor_outputs)
        self.predictor.compile('adagrad', loss_predictor(max_val))
        # connect GAN
        operations_adv = self.generator(generator_inputs)
        operations_adv = Lambda(
            drop_last_value(output_length, batch_size),
            name='adversarial_drop_last_value')(operations_adv)
        operations_adv = self.predictor(operations_adv)
        self.adversarial = Model(generator_inputs, operations_adv, name='adversarial')
        self.adversarial.compile('adagrad', loss_adv(loss_predictor(max_val)))

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

    def get_model(self) -> (Model, Model, Model):
        return self.generator, self.predictor, self.adversarial