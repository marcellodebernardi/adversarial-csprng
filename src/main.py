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

import utils
import train
from keras import Model
from keras.layers import Input, Dense, SimpleRNN, LSTM, Lambda
from keras.optimizers import sgd
from models.activations import modular_activation
from models.operations import drop_last_value
from models.losses import loss_disc, loss_gan, loss_pnb


SEED_LENGTH = 1             # the number of individual values in the seed
MAX_SEED_VAL = 10           # the max bound for each value in the seed
OUTPUT_LENGTH = 2           # the number of values outputted by the generator
BATCH_SIZE = 1              # the number of inputs in one batch
EPOCHS = 100                # epochs for training
USE_ORACLE = False


def main():
    """Instantiates neural networks and runs the training procedure. The results
    are plotted visually."""
    seed = utils.get_seed_decimal(MAX_SEED_VAL, SEED_LENGTH)
    print("Generated seed " + str(seed[0]) + " ...")

    #  create nets
    gen, disc, gan = define_networks()
    gan.summary()

    # train
    losses = train.train_gan(gan, gen, disc, seed, BATCH_SIZE, OUTPUT_LENGTH, EPOCHS)
    # print("Training losses: ")
    # print(losses)

    # plot results
    utils.plot_loss(losses['generator'], losses['discriminator'])

    # save configuration
    gan.save_weights('../saved_models/placeholder.h5', overwrite=True)
    # gan.get_model().save('../saved_models/placeholder.h5', overwrite=True)
    # save model with model.to_json, model.save, model.save_weights


def define_networks() -> (Model, Model, Model):
    """Returns the Keras models defining the generative adversarial network.
    The first model returned is the generator, the second is the discriminator,
    and the third is the connected GAN."""
    inputs_gen = Input(shape=(SEED_LENGTH,))
    operations_gen = Dense(OUTPUT_LENGTH, activation=modular_activation(50))(inputs_gen)
    gen = Model(inputs_gen, operations_gen)
    gen.compile(optimizer=sgd(lr=0.0001, clipvalue=0.5), loss=loss_pnb(OUTPUT_LENGTH))

    inputs_disc = Input(shape=(OUTPUT_LENGTH - 1,))
    operations_disc = Dense(OUTPUT_LENGTH, activation=modular_activation(50))(inputs_disc)
    operations_disc = Dense(1, )(operations_disc)
    disc = Model(inputs_disc, operations_disc)
    disc.compile(sgd(lr=0.0001, clipvalue=0.5), loss=loss_disc)

    operations_gan = gen(inputs_gen)
    operations_gan = Lambda(drop_last_value(original_size=OUTPUT_LENGTH, batch_size=BATCH_SIZE))(operations_gan)
    operations_gan = disc(operations_gan)
    gan = Model(inputs_gen, operations_gan)
    gan.compile(optimizer=sgd(lr=0.0001, clipvalue=0.5), loss=loss_gan)

    return gen, disc, gan


if __name__ == "__main__":
    main()
