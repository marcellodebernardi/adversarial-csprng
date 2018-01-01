import random
import tensorflow as tf
import numpy as np
from keras import Model
from keras.utils import plot_model


# todo are repetitions in the dataset necessary since training has epochs anyway?
def get_seed_dataset(max_seed: int, seed_size: int, unique_seeds: int, repetitions: int, batch_size=1) -> np.ndarray:
    """Returns a seed dataset for training. Each individual seed consists of
    n = seed_size real numbers in the range [0 - max_seed]. The dataset contains
    k = (unique_seeds * repetitions) seeds, split into batches of size batch_size.
    The default batch size of 1 results in a dataset suitable for online training.
    """
    # check for bad input
    if (unique_seeds * repetitions) % batch_size != 0:
        raise ValueError('The product (unique_seeds * repetitions) must be a multiple of the batch size')
    # generate unique seeds
    seeds = [[random.uniform(0, max_seed) for i in range(seed_size)] for j in range(unique_seeds)]
    # expand to include repetition of unique seeds
    seeds = np.array([seed for seed in seeds for i in range(repetitions)], dtype=np.float64)
    # split into batches
    if unique_seeds == 1 and repetitions == 1:
        return np.expand_dims(seeds, 0)
    else:
        return np.array(np.split(seeds, int(len(seeds) / batch_size)), dtype=np.float64)


def split_generator_output(generator_output: np.ndarray, n_to_predict) -> (np.ndarray, np.ndarray):
    """Takes the generator output as a numpy array and splits it into two
    separate numpy arrays, the first representing the input to the predictor
    and the second representing the output labels for the predictor."""
    batch_len = len(generator_output)
    seq_len = len(generator_output[0])
    predictor_inputs = generator_output[0: batch_len, 0: -n_to_predict]
    predictor_outputs = generator_output[0: batch_len, seq_len - n_to_predict - 1: seq_len - n_to_predict]
    return predictor_inputs, predictor_outputs


def set_trainable(model: Model, trainable: bool=True):
    """Helper method that sets the trainability of all of a model's
    parameters."""
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def log(x, base) -> tf.Tensor:
    """Allows computing element-wise logarithms on a Tensor, in
    any base. TensorFlow itself only has a natural logarithm
    operation."""
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator


def flatten_irregular_nested_iterable(weight_matrix) -> list:
    """Allows flattening a matrix of iterables where the specific type
    and shape of each iterable is not necessarily the same. Returns
    the individual elements of the original nested iterable in a single
    flat list.
    """
    flattened_list = []
    try:
        for element in weight_matrix:
            flattened_list.extend(flatten_irregular_nested_iterable(element))
        return flattened_list
    except TypeError:
        return weight_matrix


def plot_network_graphs(gen: Model, pred: Model, adv: Model):
    """Draws visualizations of the network structure as well as the
    shape of each layer."""
    plot_model(gen, to_file='../model_graphs/generator.png', show_shapes=True)
    plot_model(pred, to_file='../model_graphs/predictor.png', show_shapes=True)
    plot_model(adv, to_file='../model_graphs/adversarial.png', show_shapes=True)
