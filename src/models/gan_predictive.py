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
from data_sources import randomness_sources as data
from keras import Model
from keras.layers import Input, Lambda, Dense, SimpleRNN, LSTM, Reshape, Flatten
from keras.activations import linear, relu
from keras.optimizers import adagrad
from tqdm import tqdm
from utils import utils
from utils import vis_utils
import numpy as np


class PredictiveGan:
    def __init__(self, dataset_size=100, max_val=100, seed_length=10, unique_seeds=100, repetitions=1, output_length=300, lr=0.1, cv=1, batch_size=1):
        self.metrics = Metrics()
        self.dataset_size = dataset_size
        self.max_val = max_val
        self.seed_length = seed_length
        self.output_length = output_length
        self.unique_seeds = unique_seeds
        self.repetitions = repetitions
        # generator
        generator_inputs = Input(shape=(seed_length,))
        generator_outputs = Dense(output_length, activation=linear)(generator_inputs)
        generator_outputs = Reshape(target_shape=(5, int(output_length/5)))(generator_outputs)
        generator_outputs = SimpleRNN(60, return_sequences=True, activation=linear)(generator_outputs)
        generator_outputs = Flatten()(generator_outputs)
        generator_outputs = Dense(output_length, activation=modulo(max_val))(generator_outputs)
        self.generator = Model(generator_inputs, generator_outputs)
        self.generator.compile('adagrad', 'binary_crossentropy')
        vis_utils.plot_network_graphs(self.generator, 'disc_generator')
        # predictor
        predictor_inputs = Input(shape=(output_length - 1,))
        predictor_outputs = Dense(output_length)(predictor_inputs)
        predictor_outputs = Reshape(target_shape=(5, int(output_length/5)))(predictor_outputs)
        predictor_outputs = LSTM(int(output_length/5), return_sequences=True, activation=linear)(predictor_outputs)
        predictor_outputs = Flatten()(predictor_outputs)
        predictor_outputs = Dense(1, activation=relu)(predictor_outputs)
        self.predictor = Model(predictor_inputs, predictor_outputs)
        self.predictor.compile(adagrad(lr=lr, clipvalue=cv), loss_predictor(max_val))
        vis_utils.plot_network_graphs(self.predictor, 'pred_adversary')
        # connect GAN
        operations_adv = self.generator(generator_inputs)
        operations_adv = Lambda(
            drop_last_value(output_length, batch_size),
            name='adversarial_drop_last_value')(operations_adv)
        operations_adv = self.predictor(operations_adv)
        self.adversarial = Model(generator_inputs, operations_adv, name='adversarial')
        self.adversarial.compile(adagrad(lr=lr, clipvalue=cv), loss_adv(loss_predictor(max_val)))
        vis_utils.plot_network_graphs(self.adversarial, 'pred_gan')

    def pretrain_predictor(self, batch_size, epochs):
        """Pre-trains the predictor network on its own. Used before the
        adversarial training of both models is started."""
        seed_dataset = data.get_seed_dataset(self.max_val, self.seed_length, self.unique_seeds, self.repetitions, batch_size)
        for epoch in range(epochs):
            epoch_pretrain_loss = []
            for generator_input in seed_dataset:
                generator_output = self.generator.predict_on_batch(generator_input)
                predictor_input, predictor_output = utils.split_generator_output(generator_output, 1)
                epoch_pretrain_loss.append(self.predictor.train_on_batch(predictor_input, predictor_output))
            self.metrics.predictor_pretrain_loss().append(np.mean(epoch_pretrain_loss))

    def train(self, batch_size, epochs, pred_mult):
        """Trains the adversarial model on the given dataset of seed values, for the
        specified number of epochs. The seed dataset must be 3-dimensional, of the form
        [batch, seed, seed_component]. Each 'batch' in the dataset can be of any size,
        including 1, allowing for online training, batch training, and mini-batch training.
        """
        seed_dataset = data.get_seed_dataset(self.max_val, self.seed_length, self.unique_seeds, self.repetitions, batch_size)
        # if len(np.shape(seed_dataset)) != 3:
        #    raise ValueError('Seed dataset has ' + str(len(np.shape(seed_dataset))) + ' dimension(s), should have 3')

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
                utils.set_trainable(self.predictor)
                for i in range(pred_mult):
                    epoch_pred_losses.append(self.predictor.train_on_batch(predictor_input, predictor_output))

                # train generator
                utils.set_trainable(self.predictor, False)
                epoch_gen_losses.append(self.adversarial.train_on_batch(generator_input, predictor_output))

            self.metrics.generator_loss().append(np.mean(epoch_gen_losses))
            self.metrics.predictor_loss().append(np.mean(epoch_pred_losses))

        self.metrics.generator_weights_final().extend(utils.flatten_irregular_nested_iterable(self.generator.get_weights()))
        self.metrics.predictor_weights_final().extend(utils.flatten_irregular_nested_iterable(self.predictor.get_weights()))

    def get_model(self) -> (Model, Model, Model):
        return self.generator, self.predictor, self.adversarial

    def get_metrics(self):
        return self.metrics
