import tensorflow as tf
import numpy as np
from keras import Model


def set_trainable(model: Model, optimizer, loss, recompile, trainable: bool = True):
    """Helper method that sets the trainability of all of a model's
    parameters."""
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
    if recompile:
        model.compile(optimizer, loss)


def split_generator_outputs_batch(generator_output: np.ndarray, n_to_predict) -> (np.ndarray, np.ndarray):
    inputs = []
    outputs = []

    for i in range(len(generator_output)):
        inp, out = split_generator_output(generator_output[i], n_to_predict)
        inputs.append(inp)
        outputs.append(out)

    return np.array(inputs), np.array(outputs)


def split_generator_output(generator_output: np.ndarray, n_to_predict) -> (np.ndarray, np.ndarray):
    """Takes the generator output as a numpy array and splits it into two
    separate numpy arrays, the first representing the input to the predictor
    and the second representing the output labels for the predictor."""
    seq_len = len(generator_output)
    predictor_inputs = generator_output[0: -n_to_predict]
    predictor_outputs = generator_output[seq_len - n_to_predict - 1: seq_len - n_to_predict]
    return predictor_inputs, predictor_outputs


def get_ith_batch(data: np.ndarray, batch: int, batch_size):
    """Returns a slice of the given array, corresponding to the ith
    batch of size batch_size."""
    return data[batch * batch_size: ((batch + 1) * batch_size) - 1]


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
