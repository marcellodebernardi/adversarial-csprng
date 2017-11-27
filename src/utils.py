import secrets
import numpy as np


def get_random_seed(seed_size: int) -> np.ndarray:
    """Returns a truly random value of given size from the entropy
    source available to the OS"""
    bitfield = [int(digit) for digit in bin(secrets.randbits(seed_size))[2:]]

    for i in range(seed_size - len(bitfield)):
        bitfield.insert(i, 0)

    return np.array([bitfield])
