import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Model
from keras.utils import plot_model


def get_seed_dataset(max_seed: int, seed_size: int, num_of_seeds: int, repetitions: int) -> np.ndarray:
    """"""
    unique_seeds = [[random.uniform(0, max_seed) for i in range(seed_size)] for j in range(num_of_seeds)]
    return np.array([seed for seed in unique_seeds for i in range(repetitions)], dtype=np.float64)


def split_into_batches(seed_dataset: np.ndarray, batch_size=1) -> np.ndarray:
    """Splits the seed dataset into batches of the given size. Raises a
    ValueError if the size of the dataset is not a multiple of the batch
    size. Default batch size is 1, which is used for online training."""
    # check for bad input
    if len(seed_dataset) % batch_size != 0:
        raise ValueError('The size of the seed dataset must be a multiple of the batch size')
    return np.array(np.split(seed_dataset, int(len(seed_dataset)/batch_size)), dtype=np.float64)


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


def plot_loss(gen_loss, disc_loss):
    ax = pd.DataFrame(
        {
            'Generative Loss': gen_loss,
            'Predictive Loss': disc_loss,
        }
    ).plot(title='Training loss')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    plt.show()


def plot_generator_outputs(outputs, data_range):
    plt.hist(outputs, bins=data_range * 2)
    plt.title('Generator Output Distribution')
    plt.xlabel('Output')
    plt.ylabel('Frequency')
    plt.show()


def plot_network_graphs(gen: Model, pred: Model, adv: Model):
    plot_model(gen, to_file='../model_graphs/generator.png', show_shapes=True)
    plot_model(pred, to_file='../model_graphs/predictor.png', show_shapes=True)
    plot_model(adv, to_file='../model_graphs/adversarial.png', show_shapes=True)


