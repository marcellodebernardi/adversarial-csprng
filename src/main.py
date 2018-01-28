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

import sys
import tensorflow as tf
from utils import utils
from utils import vis_utils
from models.gan_discriminative import DiscriminativeGan
from models.gan_predictive import PredictiveGan
from evaluators import pnb


INTERACTIVE = False         # if true outputs matplotlib graphs
TESTING = True              # if true uses small parameters for rapid tests
SETTINGS = {
    'dataset_size': 4096 * 100 if not TESTING else 2,
    'unique_seeds': 128 if not TESTING else 2,
    'seed_repetitions': 8 if not TESTING else 1,
    'pretrain': True,
    'batch_size': 4096 if not TESTING else 1,
    'epochs': 200000 if not TESTING else 5,
    'pretrain_epochs': 10000 if not TESTING else 2,
    'adversary_multiplier': 2,
    'clip_value': 0.5,
    'learning_rate': 0.2
}
DATA_PARAMS = {
    'max_val': sys.maxsize if not TESTING else 100,
    'seed_length': 1,
    'out_seq_length': 5000 if not TESTING else 10
}


def main():
    """Instantiates neural networks and runs the training procedure. Results
    are plotted visually."""
    tf.logging.set_verbosity(tf.logging.ERROR)
    # command line arguments
    global INTERACTIVE
    global TESTING
    INTERACTIVE = "-i" in sys.argv
    TESTING = "-t" in sys.argv

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

    pred_gan.pretrain_predictor(SETTINGS['batch_size'],
                                SETTINGS['pretrain_epochs'])
    pred_gan.train(SETTINGS['batch_size'],
                   SETTINGS['epochs'],
                   SETTINGS['adversary_multiplier'])

    if INTERACTIVE:
        vis_utils.plot_metrics(disc_gan.get_metrics(), DATA_PARAMS['max_val'])
        vis_utils.plot_metrics(pred_gan.get_metrics(), DATA_PARAMS['max_val'])

    # generate output files, save configuration and send training report
    disc_gan.generate_output_file()
    pred_gan.generate_output_file()
    utils.save_configurations(disc_gan, pred_gan)
    utils.email_report(disc_gan.get_metrics(), pred_gan.get_metrics())
    pnb.evaluate('../sequences/disc_sequence.txt')


if __name__ == "__main__":
    main()
