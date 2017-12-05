import secrets
import tensorflow as tf


def get_random_seed(seed_size: int) -> list:
    """Returns a truly random value of given size from the entropy
    source available to the OS"""
    bitfield = [int(digit) for digit in bin(secrets.randbits(seed_size))[2:]]

    for i in range(seed_size - len(bitfield)):
        bitfield.insert(i, 0)

    return bitfield


def log(x, base) -> tf.Tensor:
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator
