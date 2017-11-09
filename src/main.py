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

# libraries
import tensorflow as tf
import os


# training parameters
SEED_LENGTH = 256
USE_ORACLES = [False, False]

# logging and housekeeping
PRINT_FREQUENCY = 200


# todo step 1: create model
# the model consists of three separate networks
