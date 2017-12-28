import utils
import numpy as np
from keras import Model


def train_gan(generator: Model, predictor: Model, adversarial: Model, seed_dataset, batch_size, epochs):
    losses = {
        'generator': [],
        'predictor': []
    }

    for epoch in range(epochs):
        for seed in seed_dataset:
            generator_input = utils.form_seed_batch(seed, batch_size)
            generator_output = generator.predict(generator_input)
            predictor_input, predictor_output = utils.split_generator_output(generator_output, batch_size, 1)

            print(generator_output.shape)
            print(predictor_input.shape)
            print(predictor_output.shape)

            utils.set_trainable(predictor)
            losses['predictor'].append(predictor.train_on_batch(predictor_input, predictor_output))

            utils.set_trainable(predictor, False)
            losses['generator'].append(adversarial.train_on_batch(generator_input, predictor_output))

    return losses
