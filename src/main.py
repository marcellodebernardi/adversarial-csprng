# Marcello De Bernardi, Queen Mary University of London
#
# A reimplementation of Abadi and Andersen's generative
# adversarial network, originally outlined in the paper
# "Learning to Protect Communications with Adversarial
# Networks", available at https://arxiv.org/abs/1610.06918.
#
# The original implementation by Abadi is available at
# https://github.com/tensorflow/models/tree/master/research/adversarial_crypto.
#
# The project seeks to improve on the original model.
# =================================================================

import tensorflow


# todo step 1: create model
# the model consists of three separate networks