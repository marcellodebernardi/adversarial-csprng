import secrets
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import email
from smtplib import SMTP


def get_seed_decimal(max_seed: int, seed_size: int) -> list:
    # todo not secure but okay for testing
    return [random.uniform(0, max_seed) for i in range(seed_size)]


def form_seed_batch(seed, batch_size=32) -> np.ndarray:
    """Returns a 2D numpy array of length batch_size, where
    each element is an array containing the values of the seed."""
    seed_batch = np.empty(shape=(batch_size, len(seed)), dtype=np.float64)
    # print(seed_batch)
    for i in range(len(seed_batch)):
        seed_batch[i] = np.array(seed, dtype=np.float64)
    return seed_batch


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
            'Discriminative Loss': disc_loss,
        }
    ).plot(title='Training loss', logy=True)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    plt.show()


def email_results():
    # SMTP.send_message()
    pass
