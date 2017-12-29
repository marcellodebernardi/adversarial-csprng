import utils
import numpy as np
from keras import Model


def train_gan(generator: Model, predictor: Model, adversarial: Model, seed_dataset, epochs):
    metrics = {
        'generator_loss': [],
        'predictor_loss': [],
        'generator_outputs': [],
        'generator_max_weight': [],
        'generator_min_weight': [],
        'generator_avg_weight': [],
        'predictor_max_weight': [],
        'predictor_min_weight': [],
        'predictor_avg_weight': [],
        'generator_final_weights': [],
        'predictor_final_weights': []
    }

    for epoch in range(epochs):
        if epoch % 100 == 0:
            print('Epoch: ' + str(epoch))
            # todo progress reporting

        # todo obtain loss for whole epoch, not for each seed to eliminate plot jitter
        for generator_input in seed_dataset:
            generator_output = generator.predict_on_batch(generator_input)
            metrics['generator_outputs'].append(generator_output)
            # todo splitting probably doesn't account for batch dimension
            predictor_input, predictor_output = utils.split_generator_output(generator_output, 1)

            # train predictor todo train multiple times
            utils.set_trainable(predictor)
            metrics['predictor_loss'].append(predictor.train_on_batch(predictor_input, predictor_output))

            # train generator
            utils.set_trainable(predictor, False)
            metrics['generator_loss'].append(adversarial.train_on_batch(generator_input, predictor_output))

            # extract metrics todo
            generator_weights = generator.get_weights()
            predictor_weights = predictor.get_weights()

    return metrics
