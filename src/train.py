import utils
import numpy as np
from keras import Model


def train_generator(gen: Model, seed, batch_size, output_length, epochs=500):
    pass


def train_gan(gan: Model, gen: Model, disc: Model, seed, batch_size, output_length, epochs=500) -> dict:
    """Performs end-to-end training of the GAN model. Returns a dictionary
    containing two lists with the loss function values for each epoch. The
    dictionary entries are labeled 'generator' and 'discriminator'."""
    losses = {
        'generator': [],
        'discriminator': []
    }
    seed_batch = utils.form_seed_batch(seed, batch_size)
    for e in range(epochs):
        # train discriminator
        set_trainable(disc, True)
        input_data, output_data = generate_disc_io_pairs(gen, seed, batch_size, output_length)
        dl = disc.train_on_batch(input_data, output_data)
        print(gan.get_weights())
        losses['discriminator'].append(dl)

        # train generator, aim is to compute loss on generated inputs
        set_trainable(disc, False)
        gl = gan.train_on_batch(seed_batch, generate_disc_io_pairs(gen, seed, batch_size, output_length)[1])
        # gl = gan.get_model().evaluate(seed_batch, generate_correct_next(gen, seed, 32, output_length))
        losses['generator'].append(gl)
    return losses


def generate_disc_io_pairs(gen: Model, seed, batch_size, output_length):
    """Generates two batches to be used as input-output pairs for training
    the discriminator model. The first value returned represents input data
    for the discriminator, while the second values represents the correct
    output data that the discriminator should output."""
    # todo prediction will probably affect state of RNN layers
    disc_input_array = np.empty((batch_size, output_length - 1))
    disc_output_array = np.empty((batch_size, 1))
    seed = np.array([seed])

    for i in range(len(disc_output_array)):
        gen_out = gen.predict(seed)
        np.append(disc_input_array, gen_out[0][0:output_length-1])
        np.append(disc_output_array, gen_out[0][output_length - 1])

    return disc_input_array, disc_output_array


def set_trainable(model: Model, trainable: bool):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable