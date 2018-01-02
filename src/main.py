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

import utils, vis_utils, train
import numpy as np
from keras import Model
from keras.layers import Input, Dense, SimpleRNN, LSTM, Lambda
from keras.activations import relu
from keras.optimizers import sgd, adagrad
from models.activations import modulo, absolute
from models.operations import drop_last_value
from models.losses import loss_predictor, loss_adv, loss_pnb
from models.metrics import Metrics

# the 'dataset' for training is a 3D matrix, where each value in
# dimension 0 is a "batch", each value in dimension 1 is a seed,
# and each value in dimension 2 is a real number that is part of
# the seed.
#
# Each epoch the model is trained on each "batch", where the batch
# may be of any size n > 0. A batch size of 1 results in online
# training, while any batch size n > 1 results in mini-batch or
# batch training. The dataset is split into "batches" regardless of
# whether online training or batch training is carried out, to
# simplify the code.

# training settings
UNIQUE_SEEDS = 400              # number of unique seeds to train with
SEED_REPETITIONS = 1            # how many times each unique seed is repeated in dataset
BATCH_MODE = True               # train in batch mode or online mode
BATCH_SIZE = 40                 # size of batch when batch training
# input/output parameters
MAX_VAL = 100                   # the max bound for each value in the seed
SEED_LENGTH = 1                 # the number of individual values in the seed
SEQ_LENGTH = 200                # the number of values outputted by the generator
# training parameters
EPOCHS = 10                     # epochs for training
PRETRAIN_EPOCHS = 10            # epochs for pre-training of the predictor
PRED_MULTIPLIER = 1             # predictor is trained more than generator
NET_CV = 1                      # clip value for networks
NET_LR = 0.0008                 # learning rate for networks


def main():
    """Instantiates neural networks and runs the training procedure. Results
    are plotted visually."""
    pretrain_seed_dataset = utils.get_seed_dataset(
        MAX_VAL,
        SEED_LENGTH,
        UNIQUE_SEEDS,
        SEED_REPETITIONS,
        BATCH_SIZE if BATCH_MODE else 1
    )
    train_seed_dataset = utils.get_seed_dataset(
        MAX_VAL,
        SEED_LENGTH,
        UNIQUE_SEEDS,
        SEED_REPETITIONS,
        BATCH_SIZE if BATCH_MODE else 1
    )
    eval_seed_dataset = utils.get_seed_dataset(
        MAX_VAL,
        SEED_LENGTH,
        UNIQUE_SEEDS * 10,
        SEED_REPETITIONS * 10,
        BATCH_SIZE if BATCH_MODE else 1
    )

    # define neural nets
    predictor, generator, adversarial = define_networks()
    utils.plot_network_graphs(generator, predictor, adversarial)

    # perform training and evaluation
    metrics = Metrics()
    train.pretrain_predictor(generator, predictor, pretrain_seed_dataset, PRETRAIN_EPOCHS, metrics)
    train.train(generator, predictor, adversarial, train_seed_dataset, EPOCHS, PRED_MULTIPLIER, metrics)
    train.evaluate(generator, adversarial, eval_seed_dataset, metrics)

    # plot results
    vis_utils.plot_metrics(metrics, MAX_VAL)

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
    operations_gen = Dense(
        SEQ_LENGTH,
        activation=modulo(MAX_VAL),
        name='generator_hidden_dense1')(inputs_gen)
    operations_gen = Dense(
        SEQ_LENGTH,
        activation=modulo(MAX_VAL),
        name='generator_output')(operations_gen)
    generator = Model(inputs_gen, operations_gen, name='generator')
    generator.compile(optimizer=adagrad(lr=NET_LR, clipvalue=NET_CV), loss='binary_crossentropy')

    # define predictor
    inputs_predictor = Input(shape=(SEQ_LENGTH - 1,), name='predictor_input')
    operations_predictor = Dense(
        SEQ_LENGTH,
        activation=absolute,
        name='predictor_hidden_dense1')(inputs_predictor)
    operations_predictor = Dense(
        1,
        activation=absolute,
        name='predictor_output')(operations_predictor)
    predictor = Model(inputs_predictor, operations_predictor, name='predictor')
    predictor.compile(adagrad(lr=NET_LR, clipvalue=NET_CV), loss=loss_predictor(MAX_VAL))

    # define adversarial model
    operations_adv = generator(inputs_gen)
    operations_adv = Lambda(
        drop_last_value(SEQ_LENGTH, UNIQUE_SEEDS if BATCH_MODE else 1),
        name='adversarial_drop_last_value')(operations_adv)
    operations_adv = predictor(operations_adv)
    adversarial = Model(inputs_gen, operations_adv, name='adversarial')
    adversarial.compile(adagrad(lr=NET_LR, clipvalue=NET_CV), loss=loss_adv(loss_predictor(MAX_VAL)))

    return predictor, generator, adversarial


if __name__ == "__main__":
    main()
