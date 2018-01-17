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
from models.gan_discriminative import DiscriminativeGan
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
    'dataset_size': 10,
    'unique_seeds': 1,
    'seed_repetitions': 1,
    'pretrain': True,
    'batch_size': 1,
    'epochs': 100,
    'pretrain_epochs': 10,
    'adversary_multiplier': 10,
    'clip_value': 1,
    'learning_rate': 0.2
}
DATA_PARAMS = {
    'max_val': 100,
    'seed_length': 1,
    'out_seq_length': 200
}


def main():
    """Instantiates neural networks and runs the training procedure. Results
    are plotted visually."""
    # approach 1: adversarial network with discriminator
    disc_gan = DiscriminativeGan(SETTINGS['dataset_size'],
                                 DATA_PARAMS['max_val'],
                                 DATA_PARAMS['seed_length'],
                                 DATA_PARAMS['out_seq_length'],
                                 SETTINGS['learning_rate'],
                                 SETTINGS['clip_value'],
                                 SETTINGS['batch_size'])
    pred_gan = PredictiveGan(SETTINGS['dataset_size'],
                             DATA_PARAMS['max_val'],
                             DATA_PARAMS['seed_length'],
                             SETTINGS['unique_seeds'],
                             SETTINGS['seed_repetitions'],
                             DATA_PARAMS['out_seq_length'],
                             SETTINGS['learning_rate'],
                             SETTINGS['clip_value'],
                             SETTINGS['batch_size'])

    disc_gan.pretrain_discriminator(SETTINGS['batch_size'],
                                    SETTINGS['pretrain_epochs'])
    disc_gan.train(SETTINGS['batch_size'],
                   SETTINGS['epochs'],
                   SETTINGS['adversary_multiplier'])
    # disc_gan.evaluate()

    pred_gan.pretrain_predictor(SETTINGS['batch_size'],
                                SETTINGS['pretrain_epochs'])
    pred_gan.train(SETTINGS['batch_size'],
                   SETTINGS['epochs'],
                   SETTINGS['adversary_multiplier'])
    # pred_gan.evaluate()

    # vis_utils.plot_metrics(classic_gan.get_metrics(), IO_PARAMS['max_val'])
    # vis_utils.plot_metrics(pred_gan.get_metrics(), IO_PARAMS['max_val'])

    # save configuration
    # gan.save_weights('../saved_models/placeholder.h5', overwrite=True)
    # gan.get_model().save('../saved_models/placeholder.h5', overwrite=True)
    # save model with model.to_json, model.save, model.save_weights


if __name__ == "__main__":
    main()
