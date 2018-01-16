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

import constants
import utils.utils as utils
from utils import vis_utils
from models.gan_classic import ClassicGan
from models.gan_predictive import PredictiveGan

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
SETTINGS = constants.SETTINGS
IO_PARAMS = constants.IO_PARAMS
TRAIN_PARAMS = constants.TRAIN_PARAMS
# network structures
GENERATOR_SPEC = constants.GENERATOR_SPEC
PREDICTOR_SPEC = constants.PREDICTOR_SPEC
DISCRIMINATOR_SPEC = constants.DISCRIMINATOR_SPEC
ADVERSARIAL_CLASSIC_SPEC = constants.ADVERSARIAL_CLASSIC_SPEC
ADVERSARIAL_SPEC = constants.ADVERSARIAL_SPEC


def main():
    """Instantiates neural networks and runs the training procedure. Results
    are plotted visually."""
    # networks for approaches 1 and 2: generator with discriminator, generator with predictor
    classic_gan = ClassicGan(GENERATOR_SPEC, DISCRIMINATOR_SPEC, ADVERSARIAL_CLASSIC_SPEC)
    pred_gan = PredictiveGan(GENERATOR_SPEC, PREDICTOR_SPEC, ADVERSARIAL_SPEC)
    vis_utils.plot_network_graphs(classic_gan, pred_gan)

    # train and evaluate approach 1
    classic_gan.pretrain_discriminator()
    classic_gan.train(TRAIN_PARAMS['epochs'], TRAIN_PARAMS['adversary_multiplier'])
    # classic_gan.evaluate()
    # train and evaluate approach 2
    pred_gan.pretrain_predictor(None, TRAIN_PARAMS['pretrain_epochs'])
    pred_gan.train(None, TRAIN_PARAMS['epochs'], TRAIN_PARAMS['adversary_multiplier'])
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
