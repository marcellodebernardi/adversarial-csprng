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
from keras.layers import Input, Dense, SimpleRNN, LSTM, Lambda
from keras.optimizers import sgd
from models.activations import modular_activation
from models.operations import drop_last_value
from models.losses import loss_disc, loss_gan
from models.network import Network


SEED_LENGTH = 1             # the number of individual values in the seed
MAX_SEED_VAL = 10           # the max bound for each value in the seed
OUTPUT_LENGTH = 64          # the number of values outputted by the generator
BATCH_SIZE = 32             # the number of inputs in one batch
EPOCHS = 10                 # epochs for training
USE_ORACLE = False


def main():
    """Run the full training and evaluation loop."""
    seed = utils.get_seed_decimal(MAX_SEED_VAL, SEED_LENGTH)

    #  create nets
    gen = create_generator()
    disc = create_discriminator()
    gan = connect_models(gen, disc)
    gan.summary()

    # train
    # print(gan.get_model().predict_on_batch(
    #    utils.form_seed_batch(seed, BATCH_SIZE)))

    losses = train.train_gan(gan, gen, disc, seed, BATCH_SIZE, OUTPUT_LENGTH, EPOCHS)

    # plot results
    utils.plot_loss(losses['generator'], losses['discriminator'])

    # save configuration
    gan.get_model().save_weights('../saved_models/placeholder.h5', overwrite=True)
    # gan.get_model().save('../saved_models/placeholder.h5', overwrite=True)
    # save model with model.to_json, model.save, model.save_weights


def create_generator() -> Network:
    """Returns a Network object encapsulating a compiled Keras model
    that represents the generator component of the GAN."""
    return Network()\
        .with_optimizer('adagrad')\
        .with_loss_function('binary_crossentropy')\
        .with_inputs(Input(shape=(SEED_LENGTH,)))\
        .add_layer(Dense(OUTPUT_LENGTH, activation='linear'))\
        .add_layer(Dense(OUTPUT_LENGTH, activation='linear'))\
        .add_layer(Dense(OUTPUT_LENGTH, activation='linear'))\
        .compile()


def create_discriminator() -> Network:
    """Returns a Network object encapsulating a compiled Keras model
    that represents the discriminator component of the GAN"""
    # optimizer = 'rmsprop',
    # loss = 'binary_crossentropy',
    # metrics = ['accuracy']
    return Network()\
        .with_optimizer(sgd(clipvalue=0.5))\
        .with_loss_function(loss_disc)\
        .with_inputs(Input(shape=(OUTPUT_LENGTH - 1,)))\
        .add_layer(Dense(OUTPUT_LENGTH, activation='linear'))\
        .add_layer(Dense(OUTPUT_LENGTH, activation='linear'))\
        .add_layer(Dense(1,))\
        .compile()


def connect_models(generator: Network, discriminator: Network) -> Network:
    """Connects the generator and discriminator models into a new Keras
    model by adding an intermediate layer between them that removes the
    last element from the output produced by the generator."""
    return Network()\
        .with_optimizer(sgd(clipvalue=0.5))\
        .with_loss_function(loss_gan)\
        .with_inputs(generator.get_input_tensor())\
        .add_layer(generator.get_model())\
        .add_layer(Lambda(drop_last_value(original_size=OUTPUT_LENGTH, batch_size=BATCH_SIZE)))\
        .add_layer(discriminator.get_model())\
        .compile()


if __name__ == "__main__":
    main()
