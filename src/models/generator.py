# Marcello De Bernardi, Queen Mary University of London
# =================================================================
import tensorflow as tf
import os
from models.network import Network


class Generator(Network):
    """Defines the generator neural network model, 'Jerry'. """

    def __init__(self, seed_length, with_oracle = False):
        super().__init__()
        # todo set specific optimizer (i.e. Adam.minimize.something)
        self.seed_length = seed_length
        self.seed = os.urandom(int(seed_length/8))

    def reseed(self):
        """Reseeds the generator"""
        self.seed = os.urandom(self.seed_length/8)

    def model(self):
        """Fully connected layer followed by 4 convolutional layers"""
