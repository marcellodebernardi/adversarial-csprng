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

import utils
import train
import numpy as np
from keras import Model
from keras.layers import Input, Dense, SimpleRNN, LSTM, Lambda
from keras.optimizers import sgd
from models.activations import modular_activation
from models.operations import drop_last_value
from models.losses import loss_disc, loss_adv, loss_pnb

# the 'dataset' for training is a 2D matrix, where each value in
# dimension 0 is a seed, and each value in dimension 1 is a number
# that is part of the seed. The size of dimension 0 is given by
# n = UNIQUE_SEEDS * SEED_REPETITIONS. Each epoch the adversarial
# model thus receives n inputs.
#
# In batch training, the networks are trained each epoch with
# SEED_REPETITIONS batches, each batch of size UNIQUE_SEEDS. In
# online training, the networks are trained with the same seeds
# but gradient updates are performed for each seed.

BATCH_MODE = True           # train in batch mode or online mode
SEED_LENGTH = 1             # the number of individual values in the seed
UNIQUE_SEEDS = 10           # number of unique seeds to train with
SEED_REPETITIONS = 50       # how many times each unique seed is repeated in dataset
MAX_VAL = 100               # the max bound for each value in the seed
SEQ_LENGTH = 20             # the number of values outputted by the generator
EPOCHS = 10                 # epochs for training
NET_CV = 0.5                # clip value for networks
NET_LR = 0.0008             # learning rate for networks


def main():
    """Instantiates neural networks and runs the training procedure. Results
    are plotted visually."""
    seed_dataset = utils.split_into_batches(
        utils.get_seed_dataset(MAX_VAL, SEED_LENGTH, UNIQUE_SEEDS, SEED_REPETITIONS),
        UNIQUE_SEEDS if BATCH_MODE else 1
    )

    # define neural nets
    predictor, generator, adversarial = define_networks()
    utils.plot_network_graphs(generator, predictor, adversarial)

    # train nets
    metrics = train.train_gan(generator, predictor, adversarial, seed_dataset, EPOCHS)

    # plot results
    utils.plot_loss(metrics['generator_loss'], metrics['predictor_loss'])
    utils.plot_generator_outputs(np.array(metrics['generator_outputs']).flatten(), MAX_VAL)

    # save configuration
    # gan.save_weights('../saved_models/placeholder.h5', overwrite=True)
    # gan.get_model().save('../saved_models/placeholder.h5', overwrite=True)
    # save model with model.to_json, model.save, model.save_weights


def define_networks() -> (Model, Model):
    """Returns the Keras models defining the generative adversarial network.
    The first model returned is the generator, the second is the discriminator,
    and the third is the connected GAN."""
    # define generator
    inputs_gen = Input(shape=(SEED_LENGTH,), name='generator_input')
    operations_gen = Dense(SEQ_LENGTH, activation=modular_activation(MAX_VAL), name='generator_hidden_dense1')(inputs_gen)
    operations_gen = Dense(SEQ_LENGTH, activation=modular_activation(MAX_VAL), name='generator_output')(operations_gen)
    generator = Model(inputs_gen, operations_gen, name='generator')
    generator.compile(optimizer=sgd(lr=NET_LR, clipvalue=NET_CV), loss='binary_crossentropy')

    # define predictor
    inputs_predictor = Input(shape=(SEQ_LENGTH - 1,), name='predictor_input')
    operations_predictor = Dense(SEQ_LENGTH, activation=modular_activation(MAX_VAL), name='predictor_hidden_dense1')(inputs_predictor)
    operations_predictor = Dense(1, activation=modular_activation(MAX_VAL), name='predictor_output')(operations_predictor)
    predictor = Model(inputs_predictor, operations_predictor, name='predictor')
    predictor.compile(sgd(lr=NET_LR, clipvalue=NET_CV), loss=loss_disc)

    # define adversarial model
    operations_adv = generator(inputs_gen)
    operations_adv = Lambda(
        drop_last_value(SEQ_LENGTH, UNIQUE_SEEDS if BATCH_MODE else 1),
        name='adversarial_drop_last_value')(operations_adv)
    operations_adv = predictor(operations_adv)
    adversarial = Model(inputs_gen, operations_adv, name='adversarial')
    adversarial.compile(sgd(lr=NET_LR, clipvalue=NET_CV), loss=loss_adv(loss_disc))

    return predictor, generator, adversarial


if __name__ == "__main__":
    main()
