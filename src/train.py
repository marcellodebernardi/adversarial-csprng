import utils
import numpy as np
from models.network import Network


def train_gan(gan: Network, gen: Network, disc: Network, seed, batch_size, output_length, epochs=500):
    """Performs end-to-end training of the GAN model."""
    seed_batch = utils.form_seed_batch(seed, batch_size)
    for e in range(epochs):
        # todo train discriminator, aim is to get discriminator to discern better

        # todo train generator, aim is to compute loss on generated inputs
        print(gan.get_model().train_on_batch(seed_batch, generate_correct_nb(gen, seed, 32, output_length)))


def train_disc(disc: Network, input_data, output_data, epochs=500):
    """Used to perform pre-training on the discriminator only."""
    # todo decide on batch size
    # todo decide how to pre-train
    disc.trainable().get_model().fit(input_data, output_data, epochs)


def generate_correct_nb(gen: Network, seed, batch_size, output_length):
    """Generates a batch of final sequence bits from the generator.
    These are used as the 'correct' values that the discriminator
    should be outputting during training."""
    # todo prediction will probably affect state of RNN layers
    final_bit_array = np.empty((batch_size, 1))
    seed = np.array([seed])
    for i in range(len(final_bit_array)):
        np.append(final_bit_array, gen.get_model().predict(seed)[0][output_length - 1])
    return final_bit_array
