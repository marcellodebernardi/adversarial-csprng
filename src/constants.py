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
Training constants and specifications for the network structures
of the networks used in this project.
"""

from activations import modulo, absolute
from losses import loss_predictor, loss_adv, loss_discriminator
from keras.optimizers import adagrad
from keras.activations import linear

SETTINGS = {
    'batch_mode': True,
    'dataset_size': 10,
    'unique_seeds': 1,
    'seed_repetitions': 1,
    'batch_size': 1
}
IO_PARAMS = {
    'max_val': 100,
    'seed_length': 1,
    'seq_length': 200
}
TRAIN_PARAMS = {
    'pretrain': True,
    'epochs': 10,
    'pretrain_epochs': 10,
    'adversary_multiplier': 10,
    'clip_value': 1,
    'learning_rate': 0.2
}
# network structures
GENERATOR_SPEC = {
    'name': 'generator',
    'seed_len': IO_PARAMS['seed_length'],
    'out_seq_len': IO_PARAMS['seq_length'],
    'depth': 3,
    'types': ['dense'],
    'activation': modulo(IO_PARAMS['max_val']),
    'loss': 'binary_crossentropy',
    'optimizer': 'adam'
}
PREDICTOR_SPEC = {
    'name': 'predictor',
    'depth': 3,
    'types': ['dense'],
    'activation': absolute,
    'loss': loss_predictor(IO_PARAMS['max_val']),
    'optimizer': adagrad(lr=TRAIN_PARAMS['learning_rate'], clipvalue=TRAIN_PARAMS['clip_value'])
}
DISCRIMINATOR_SPEC = {
    'name': 'discriminator',
    'depth': 3,
    'types': ['dense'],
    'activation': linear,
    'loss': 'binary_crossentropy',
    'optimizer': adagrad(lr=TRAIN_PARAMS['learning_rate'], clipvalue=TRAIN_PARAMS['clip_value'])
}
ADVERSARIAL_CLASSIC_SPEC = {
    'name': 'adversarial_classic',
    'loss': loss_adv(loss_discriminator),
    'optimizer': adagrad(lr=TRAIN_PARAMS['learning_rate'], clipvalue=TRAIN_PARAMS['clip_value'])
}
ADVERSARIAL_SPEC = {
    'name': 'adversarial',
    'loss': loss_adv(loss_predictor(IO_PARAMS['max_val'])),
    'optimizer': adagrad(lr=TRAIN_PARAMS['learning_rate'], clipvalue=TRAIN_PARAMS['clip_value'])
}
