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
from models.losses import loss_disc, loss_adv, loss_pnb


SEED_LENGTH = 1             # the number of individual values in the seed
DATASET_SIZE = 10           # number of seeds to train with
MAX_VAL = 100               # the max bound for each value in the seed
SEQ_LENGTH = 10             # the number of values outputted by the generator
BATCH_SIZE = 10             # the number of inputs in one batch
EPOCHS = 100                # epochs for training
NET_CV = 0.5                # clip value for networks
NET_LR = 0.0008             # learning rate for networks


def main():
    """Instantiates neural networks and runs the training procedure. The results
    are plotted visually."""
    seed_dataset = utils.get_seed_dataset(MAX_VAL, SEED_LENGTH, DATASET_SIZE)

    # define neural nets
    predictor, generator, adversarial = define_networks()
    utils.plot_network_graphs(generator, predictor, adversarial)

    # train nets
    # losses = train.train_gan(gan, generator, predictor, seed, BATCH_SIZE, SEQ_LENGTH, EPOCHS)
    losses = train.train_gan(generator, predictor, adversarial, seed_dataset, BATCH_SIZE, EPOCHS)

    # plot results
    utils.plot_loss(losses['generator'], losses['predictor'])

    # save configuration
    # gan.save_weights('../saved_models/placeholder.h5', overwrite=True)
    # gan.get_model().save('../saved_models/placeholder.h5', overwrite=True)
    # save model with model.to_json, model.save, model.save_weights


def define_networks() -> (Model, Model):
    """Returns the Keras models defining the generative adversarial network.
    The first model returned is the generator, the second is the discriminator,
    and the third is the connected GAN."""
    inputs_gen = Input(shape=(SEED_LENGTH,), name='generator_input')
    operations_gen = Dense(SEQ_LENGTH, activation=modular_activation(MAX_VAL), name='generator_output')(inputs_gen)
    generator = Model(inputs_gen, operations_gen, name='generator')
    generator.compile(optimizer=sgd(lr=NET_LR, clipvalue=NET_CV), loss='binary_crossentropy')

    inputs_predictor = Input(shape=(SEQ_LENGTH - 1,), name='predictor_input')
    operations_predictor = Dense(SEQ_LENGTH, activation=modular_activation(MAX_VAL), name='predictor_hidden_dense1')(inputs_predictor)
    operations_predictor = Dense(1, activation=modular_activation(MAX_VAL), name='predictor_output')(operations_predictor)
    predictor = Model(inputs_predictor, operations_predictor, name='predictor')
    predictor.compile(sgd(lr=NET_LR, clipvalue=NET_CV), loss=loss_disc)

    operations_adv = generator(inputs_gen)
    operations_adv = Lambda(drop_last_value(SEQ_LENGTH, BATCH_SIZE), name='adversarial_drop_last_value')(operations_adv)
    operations_adv = predictor(operations_adv)
    adversarial = Model(inputs_gen, operations_adv, name='adversarial')
    adversarial.compile(sgd(lr=NET_LR, clipvalue=NET_CV), loss=loss_adv(loss_disc))

    return predictor, generator, adversarial


if __name__ == "__main__":
    main()
