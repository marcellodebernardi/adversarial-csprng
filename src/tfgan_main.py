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


import tensorflow as tf
import tensorflow.contrib.gan as tfgan

# training parameters
batch_size = 10
learning_rate = 0.0008
seed_length = 256
use_oracles = [False, False]
max_training_cycles = 10000
iters_per_actor = 1
discriminator_multiplier = 2
# logging and housekeeping
print_frequency = 200


def main():
    """Run the full training and evaluation loop"""

    # Set up the input.
    images = tf.random_uniform([batch_size, seed_length])
    noise = tf.random_uniform([batch_size, seed_length])

    # Build the generator and discriminator.
    gan_model = tfgan.gan_model(
        generator_fn=tf.matmul(images, noise),
        discriminator_fn=tf.matmul(images, noise),  # you define
        real_data=images,
        generator_inputs=noise)

    # Build the GAN loss.
    gan_loss = tfgan.gan_loss(
        gan_model,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss)

    # Create the train ops, which calculate gradients and apply updates to weights.
    train_ops = tfgan.gan_train_ops(
        gan_model,
        gan_loss,
        generator_optimizer=tf.train.AdamOptimizer(learning_rate),
        discriminator_optimizer=tf.train.AdamOptimizer(learning_rate))

    # Run the train ops in the alternating training scheme.
    tfgan.gan_train(
        train_ops,
        hooks=[tf.train.StopAtStepHook(num_steps=max_training_cycles)],
        logdir="./")


if __name__ == "__main__":
    main()
