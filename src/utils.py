import secrets
import tensorflow as tf
import numpy as np
import email
from smtplib import SMTP


def get_random_seed(seed_size: int) -> list:
    """Returns a truly random value of given size from the entropy
    source available to the OS"""
    bitfield = [int(digit) for digit in bin(secrets.randbits(seed_size))[2:]]
    for i in range(seed_size - len(bitfield)):
        bitfield.insert(i, 0)
    return bitfield


def form_seed_batch(seed, batch_size=32) -> np.ndarray:
    """Returns a 2D numpy array of length batch_size, where
    each element is an array containing the bits of the seed."""
    seed_batch = np.empty(shape=(batch_size, len(seed)), dtype=np.int32)
    # print(seed_batch)
    for i in range(len(seed_batch)):
        seed_batch[i] = np.array(seed, dtype=np.int32)
    return seed_batch


def log(x, base) -> tf.Tensor:
    """Allows computing element-wise logarithms on a Tensor, in
    any base. TensorFlow itself only has a natural logarithm
    operation."""
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator


def email_results():
    # SMTP.send_message()
    pass
