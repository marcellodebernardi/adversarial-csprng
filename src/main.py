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
from models.activations import modulo, absolute
from models.losses import loss_predictor, loss_adv
from utils import vis_utils
from models.gan_classic import ClassicGan
from models.gan_predictive import PredictiveGan
from keras.losses import binary_crossentropy
from keras.activations import linear, softmax
from keras.optimizers import adagrad

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
# todo get rid of depth as it is implicit in length of types list
GENERATOR_SPEC = {
    'name': 'generator',
    'seed_len': IO_PARAMS['seed_length'],
    'out_seq_len': IO_PARAMS['seq_length'],
    'types': ['dense', 'dense', 'dense'],
    'activations': [modulo(IO_PARAMS['max_val']), modulo(IO_PARAMS['max_val']), modulo(IO_PARAMS['max_val'])],
    'loss': 'binary_crossentropy',
    'optimizer': 'adam'
}
PREDICTOR_SPEC = {
    'name': 'predictor',
    'types': ['dense', 'dense', 'dense'],
    'activations': [absolute, absolute, absolute],
    'loss': loss_predictor(IO_PARAMS['max_val']),
    'optimizer': adagrad(lr=TRAIN_PARAMS['learning_rate'], clipvalue=TRAIN_PARAMS['clip_value'])
}
DISCRIMINATOR_SPEC = {
    'name': 'discriminator',
    'types': ['dense', 'dense', 'dense'],
    'activations': [linear, linear, softmax],
    'loss': binary_crossentropy,
    'optimizer': adagrad(lr=TRAIN_PARAMS['learning_rate'], clipvalue=TRAIN_PARAMS['clip_value'])
}
ADVERSARIAL_CLASSIC_SPEC = {
    'name': 'adversarial_classic',
    'loss': loss_adv(binary_crossentropy),
    'optimizer': adagrad(lr=TRAIN_PARAMS['learning_rate'], clipvalue=TRAIN_PARAMS['clip_value'])
}
ADVERSARIAL_SPEC = {
    'name': 'adversarial',
    'loss': loss_adv(loss_predictor(IO_PARAMS['max_val'])),
    'optimizer': adagrad(lr=TRAIN_PARAMS['learning_rate'], clipvalue=TRAIN_PARAMS['clip_value'])
}


def main():
    """Instantiates neural networks and runs the training procedure. Results
    are plotted visually."""
    # networks for approaches 1 and 2: generator with discriminator, generator with predictor
    classic_gan = ClassicGan(
        GENERATOR_SPEC,
        DISCRIMINATOR_SPEC,
        ADVERSARIAL_CLASSIC_SPEC,
        SETTINGS,
        IO_PARAMS,
        TRAIN_PARAMS)
    classic_gan.get_generator().summary()
    classic_gan.get_discriminator().summary()
    pred_gan = PredictiveGan(
        GENERATOR_SPEC,
        PREDICTOR_SPEC,
        ADVERSARIAL_SPEC,
        SETTINGS,
        IO_PARAMS,
        TRAIN_PARAMS
    )
    pred_gan.get_generator().summary()
    pred_gan.get_predictor().summary()
    vis_utils.plot_network_graphs(classic_gan, pred_gan)

    # train and evaluate approach 1
    classic_gan.pretrain_discriminator(TRAIN_PARAMS['pretrain_epochs'])
    # classic_gan.train(TRAIN_PARAMS['epochs'], TRAIN_PARAMS['adversary_multiplier'])
    # classic_gan.evaluate()
    # train and evaluate approach 2
    # pred_gan.pretrain_predictor(None, TRAIN_PARAMS['pretrain_epochs'])
    # pred_gan.train(None, TRAIN_PARAMS['epochs'], TRAIN_PARAMS['adversary_multiplier'])
    # pred_gan.evaluate()

    # plot results
    vis_utils.plot_metrics(classic_gan.get_metrics(), IO_PARAMS['max_val'])
    vis_utils.plot_metrics(pred_gan.get_metrics(), IO_PARAMS['max_val'])

    # save configuration
    # gan.save_weights('../saved_models/placeholder.h5', overwrite=True)
    # gan.get_model().save('../saved_models/placeholder.h5', overwrite=True)
    # save model with model.to_json, model.save, model.save_weights


if __name__ == "__main__":
    main()
