import utils
import numpy as np
from keras import Model
from models.metrics import Metrics


def train(generator: Model, predictor: Model, adversarial: Model, seed_dataset, epochs, metrics: Metrics):
    """Trains the adversarial model on the given dataset of seed values, for the
    specified number of epochs. The seed dataset must be 3-dimensional, of the form
    [batch, seed, seed_component]. Each 'batch' in the dataset can be of any size,
    including 1, allowing for online training, batch training, and mini-batch training.
    """
    if len(np.shape(seed_dataset)) != 3:
        raise ValueError('Seed dataset has length ' + str(len(np.shape(seed_dataset))) + ', should be 3')

    # each epoch train on entire dataset
    for epoch in range(epochs):
        # todo progress reporting
        # todo obtain loss for entire epoch to eliminate plot jitter

        # the length of generator input determines whether training
        # is effectively batch training, mini-batch training or
        # online training. This is a property of the dataset
        # todo should not be a property of the dataset
        for generator_input in seed_dataset:
            generator_output = generator.predict_on_batch(generator_input)
            metrics.generator_outputs().extend(generator_output.flatten())
            metrics.generator_avg_outputs().append(np.mean(generator_output.flatten()))

            predictor_input, predictor_output = utils.split_generator_output(generator_output, 1)

            # train predictor todo train multiple times
            utils.set_trainable(predictor)
            metrics.predictor_loss().append(predictor.train_on_batch(predictor_input, predictor_output))

            # train generator
            utils.set_trainable(predictor, False)
            metrics.generator_loss().append(adversarial.train_on_batch(generator_input, predictor_output))

            # extract metrics todo
            generator_weights = generator.get_weights()
            predictor_weights = predictor.get_weights()
    return metrics


def evaluate(generator: Model, seed_dataset, metrics: Metrics):
    for batch in seed_dataset:
        metrics.generator_eval_outputs().extend(generator.predict_on_batch(batch).flatten())