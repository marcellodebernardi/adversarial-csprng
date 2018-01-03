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
This module provides utility methods for the instantiation and manipulation
of artificial neural networks implemented as Keras models. It *does not* add
meaningful functionality beyond providing a parameterized interface to the
compilation of the neural network system used in this project.

The implementation details of the methods in this module are of no real interest
for understanding the functioning of the system. It is sufficient to understand
that the methods in the module return compiled Keras models with the attributes
specified in the method call arguments.
"""

from utils.utils import eprint
from keras import Model
from keras.layers import Input, Dense, Dropout, Lambda, SimpleRNN, GRU, LSTM, ConvLSTM2D, Conv1D, Conv2D
from operations import drop_last_value


def set_trainable(model: Model, trainable: bool=True):
    """Helper method that sets the trainability of all of a model's
    parameters."""
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def construct_predictive_gan(seed_len, output_len, batch_size, gen_spec: dict, pred_spec: dict, adv_spec: dict) -> (Model, Model, Model):
    """Constructs a generator network according to the specified parameters. The form of the
    specification dictionaries is as follows:
    gen_spec, pred_spec = {
        'depth': int,
        'types': [str],
        'activation': <Keras activation function>,
        'loss': <Keras loss function>,
        'optimizer': <Keras optimizer function>
    }
    adv_spec = {
        'loss': <Keras loss function>,
        'optimizer': <Keras optimizer function>
    }
    """
    # nets
    inputs_gen, generator = __generator(seed_len, output_len, gen_spec)
    predictor = __predictor(output_len, pred_spec)

    # adversarial connection
    operations_adv = generator(inputs_gen)
    operations_adv = Lambda(
        drop_last_value(output_len, batch_size),
        name='adversarial_drop_last_value')(operations_adv)
    operations_adv = predictor(operations_adv)
    adversarial = Model(inputs_gen, operations_adv, name='adversarial')
    adversarial.compile(adv_spec['optimizer'], adv_spec['loss'])

    return generator, predictor, adversarial


def construct_discriminative_gan(seed_len, output_len, gen_spec: dict, disc_spec: dict, adv_spec: dict) -> (Model, Model, Model):
    """Constructs a generator network according to the specified parameters. The form of the
    specification dictionaries is as follows:
    gen_spec, disc_spec = {
        'name': name,
        'depth': int,
        'types': [str],
        'activation': <Keras activation function>,
        'loss': <Keras loss function>,
        'optimizer': <Keras optimizer function>
    }
    adv_spec = {
        'name': name,
        'loss': <Keras loss function>,
        'optimizer': <Keras optimizer function>
    }
    """
    # nets
    inputs_gen, generator = __generator(seed_len, output_len, gen_spec)
    discriminator = __discriminator()

    # adversarial connection
    operations_adv = generator(inputs_gen)
    operations_adv = discriminator(operations_adv)
    adversarial = Model(inputs_gen, operations_adv, name='adversarial')
    adversarial.compile(adv_spec['optimizer'], adv_spec['loss'])

    return generator, discriminator, adversarial


def construct_discriminator(name, depth, types: list, input_len, activation, optimizer, loss) -> Model:
    """Constructs a discriminator network according to the specified parameters."""
    # check bad input
    __check_parameters(depth, types)

    # input
    inputs_disc = Input(shape=(input_len,), name=str(name) + '_input')

    # operations
    operations_disc = __layer(types[0], input_len)(inputs_disc)
    for layer_index in range(1, depth):
        type_index = layer_index if layer_index < len(types) else len(types) - 1
        operations_disc = __layer(types[type_index], input, activation)(operations_disc)

    # compile model
    discriminator = Model(inputs_disc, operations_disc, name=name)
    discriminator.compile(optimizer, loss)
    return discriminator


def __generator(seed_len, output_len, gen_spec):
    """Returns a compiled generator model"""
    __check_parameters(gen_spec)

    inputs_gen = Input(shape=(seed_len,))
    operations_gen = __layer(gen_spec['types'][0], output_len, gen_spec['activation'])(inputs_gen)
    for layer_index in range(1, gen_spec['depth']):
        type_index = layer_index if layer_index < len(gen_spec['types']) else len(gen_spec['types']) - 1
        operations_gen = __layer(gen_spec['types'][type_index], output_len, gen_spec['activation'])(operations_gen)
    generator = Model(inputs_gen, operations_gen, name='generator')
    generator.compile(gen_spec['optimizer'], gen_spec['loss'])

    return inputs_gen, generator


def __predictor(output_len, pred_spec):
    """Returns a compiled predictor model"""
    # check bad input
    __check_parameters(pred_spec)

    inputs_pred = Input(shape=(output_len - 1,))
    operations_pred = __layer(pred_spec['types'][0], output_len if pred_spec['depth'] > 1 else 1, pred_spec['activation'])(inputs_pred)
    for layer_index in range(1, pred_spec['depth']):
        type_index = layer_index if layer_index < len(pred_spec['types']) else len(pred_spec['types']) - 1
        operations_pred = __layer(pred_spec['types'][type_index], output_len if layer_index < pred_spec['depth'] - 1 else 1, pred_spec['activation'])(operations_pred)
    predictor = Model(inputs_pred, operations_pred, name='predictor')
    predictor.compile(pred_spec['optimizer'], pred_spec['loss'])

    return predictor


def __discriminator(output_len, disc_spec) -> Model:
    """Returns a compiled discriminator model"""
    __check_parameters(disc_spec)

    inputs_disc = Input(shape=(output_len - 1,))
    operations_disc = __layer(disc_spec['types'][0], output_len - 1, disc_spec['activation'])(inputs_disc)
    for layer_index in range(1, disc_spec['depth']):
        type_index = layer_index if layer_index < len(disc_spec['types']) else len(disc_spec['types']) - 1
        operations_disc = __layer(disc_spec['types'][type_index], output_len, disc_spec['activation'])(operations_disc)
    discriminator = Model(inputs_disc, operations_disc, name='discriminator')
    discriminator.compile(disc_spec['optimizer'], disc_spec['loss'])

    return discriminator


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


def __check_parameters(specification):
    """Checks whether the network definition parameters are correct, and prints
    warnings for parameters that are usable, but likely to be incorrect."""
    if specification['depth'] < 1:
        raise ValueError('Model depth must be at least 1')
    if len(specification['types']) < 1:
        raise ValueError('Types list is empty, must contain at least one type')
    elif len(specification['types']) > specification['depth']:
        eprint('Warning: ' + specification['name'] + ' number of layer types exceeds network depth.')
    elif len(specification['types']) < specification['depth']:
        eprint('Warning: ' + specification['name'] + ' number of layer types is less than network depth.')
