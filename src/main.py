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
from models.discriminator import Discriminator
from models.generator import Generator


# training parameters
SEED_LENGTH = 256
USE_ORACLES = [False, False]
MAX_TRAINING_CYCLES = 10000
ITERS_PER_ACTOR = 1
DISCRIMINATOR_MULTIPLIER = 2
# logging and housekeeping
PRINT_FREQUENCY = 200


def main():
    """Run the full training and evaluation loop"""

    generator = Generator(SEED_LENGTH)
    discriminator = Discriminator()
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        for i in range(MAX_TRAINING_CYCLES):
            for j in range(ITERS_PER_ACTOR):
                generator.with_session(session).optimize()
            for j in range(ITERS_PER_ACTOR * DISCRIMINATOR_MULTIPLIER):
                discriminator.with_session(session).optimize()

            if i % PRINT_FREQUENCY == 0:
                generator_loss = generator.evaluate()
                discriminator_loss = discriminator.evaluate()


if __name__ == "__main__":
    main()
