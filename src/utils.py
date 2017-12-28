import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Model
from keras.utils import plot_model


def get_seed_dataset(max_seed: int, seed_size: int, num_of_seeds: int) -> list:
    """Returns a list of length seed_size, consisting of non-cryptographic
    pseudo-random numbers in the range [0-max_seed]."""
    return [[random.uniform(0, max_seed) for i in range(seed_size)] for j in range(num_of_seeds)]


def form_seed_batch(seed: list, batch_size=32) -> np.ndarray:
    """Returns a 2D numpy array of length batch_size, where
    each element is a seed array. That is, creates an array
    containing batch_size duplicates of the seed array passed
    as argument."""
    seed_batch = np.empty(shape=(batch_size, len(seed)), dtype=np.float64)
    # print(seed_batch)
    for i in range(len(seed_batch)):
        seed_batch[i] = np.array(seed, dtype=np.float64)
    return seed_batch


def split_generator_output(generator_output: np.ndarray, batch_size, n_to_predict) -> (np.ndarray, np.ndarray):
    """Takes the generator output as a numpy array and splits it into two
    separate numpy arrays, the first representing the input to the predictor
    and the second representing the output labels for the predictor."""
    predictor_inputs = generator_output[0: batch_size, 0: -n_to_predict]
    predictor_outputs = generator_output[0: batch_size, batch_size - n_to_predict - 1: batch_size - n_to_predict]
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


def plot_network_graphs(gen: Model, disc: Model, adv: Model):
    plot_model(gen, to_file='../model_graphs/generator.png', show_shapes=True)
    plot_model(disc, to_file='../model_graphs/discriminator.png', show_shapes=True)
    plot_model(adv, to_file='../model_graphs/adversarial.png', show_shapes=True)