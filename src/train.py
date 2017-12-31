import utils
import numpy as np
from keras import Model
from tqdm import tqdm
from models.metrics import Metrics


def train(generator: Model, predictor: Model, adversarial: Model, seed_dataset, epochs, metrics: Metrics):
    """Trains the adversarial model on the given dataset of seed values, for the
    specified number of epochs. The seed dataset must be 3-dimensional, of the form
    [batch, seed, seed_component]. Each 'batch' in the dataset can be of any size,
    including 1, allowing for online training, batch training, and mini-batch training.
    """
    if len(np.shape(seed_dataset)) != 3:
        raise ValueError('Seed dataset has ' + str(len(np.shape(seed_dataset))) + ' dimension(s), should have 3')

    # each epoch train on entire dataset
    for epoch in tqdm(range(epochs), desc='Training: '):
        # todo obtain loss for entire epoch to eliminate plot jitter

        epoch_gen_losses = []
        epoch_pred_losses = []
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
            epoch_pred_losses.append(predictor.train_on_batch(predictor_input, predictor_output))

            # train generator
            utils.set_trainable(predictor, False)
            epoch_gen_losses.append(adversarial.train_on_batch(generator_input, predictor_output))

            # extract metrics todo
            generator_weights = generator.get_weights()
            predictor_weights = predictor.get_weights()

        metrics.generator_loss().append(np.mean(epoch_gen_losses))
        metrics.predictor_loss().append(np.mean(epoch_pred_losses))

    return metrics


def evaluate(generator: Model, adversarial: Model, seed_dataset, metrics: Metrics):
    for generator_input in tqdm(seed_dataset, desc='Evaluating: '):
        generator_output = generator.predict_on_batch(generator_input)
        metrics.generator_eval_outputs().extend(generator_output.flatten())
