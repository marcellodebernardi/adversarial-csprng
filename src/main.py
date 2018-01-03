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

import utils.utils as utils
import utils.nn_utils as nn_utils
import utils.vis_utils as vis_utils
import train
from utils import vis_utils
from keras import Model
from keras.layers import Input, Dense, Lambda
from keras.optimizers import adagrad
from keras.activations import linear
from models.activations import modulo, absolute
from models.operations import drop_last_value
from models.losses import loss_predictor, loss_discriminator, loss_adv
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
BATCH_MODE = True               # train in batch mode or online mode
UNIQUE_SEEDS = 1                # number of unique seeds to train with
SEED_REPETITIONS = 1            # how many times each unique seed is repeated in dataset
BATCH_SIZE = 1                  # size of batch when batch training
# input/output parameters
MAX_VAL = 100                   # the max bound for each value in the seed
SEED_LENGTH = 1                 # the number of individual values in the seed
SEQ_LENGTH = 200                # the number of values outputted by the generator
# training parameters
PRETRAIN_PREDICTOR = True       # if true, predictor is trained alone before adversarial training
EPOCHS = 1000                   # epochs for training
PRETRAIN_EPOCHS = 10            # epochs for pre-training of the predictor
PRED_MULTIPLIER = 10            # predictor is trained more than generator
NET_CV = 1                      # clip value for networks
NET_LR = 0.2                    # learning rate for networks
# network structures
GENERATOR_SPEC = {
    'name': 'generator',
    'depth': 3,
    'types': ['dense'],
    'activation': modulo(MAX_VAL),
    'loss': 'binary_crossentropy',
    'optimizer': 'adam'
}
PREDICTOR_SPEC = {
    'name': 'predictor',
    'depth': 3,
    'types': ['dense'],
    'activation': absolute,
    'loss': loss_predictor(MAX_VAL),
    'optimizer': adagrad(lr=NET_LR, clipvalue=NET_CV)
}
DISCRIMINATOR_SPEC = {
    'name': 'discriminator',
    'depth': 3,
    'types': ['dense'],
    'activation': linear,
    'loss': loss_discriminator,
    'optimizer': adagrad(lr=NET_LR, clipvalue=NET_CV)
}
ADVERSARIAL_SPEC = {
    'name': 'adversarial',
    'loss': loss_adv(loss_predictor(MAX_VAL)),
    'optimizer': adagrad(lr=NET_LR, clipvalue=NET_CV)
}


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

    # construct networks, construction procedure abstracted by nn_utils
    generator, predictor, adversarial = nn_utils.construct_predictive_gan(
        SEED_LENGTH,
        SEQ_LENGTH,
        BATCH_SIZE,
        GENERATOR_SPEC,
        PREDICTOR_SPEC,
        ADVERSARIAL_SPEC
    )
    vis_utils.plot_network_graphs(generator, predictor, adversarial)

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


if __name__ == "__main__":
    main()
