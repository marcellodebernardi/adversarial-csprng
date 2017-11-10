# Marcello De Bernardi, Queen Mary University of London
# =================================================================
import tensorflow as tf
from models.network import Network


class Discriminator(Network):
    """Defines the discriminator neural network model, 'Eve'."""

    def __init__(self, with_oracle = False):
        super().__init__()
        # todo set specific optimizer (i.e. Adam.minimize.something)
        self.session = None
        self.optimizer = tf.train.AdamOptimizer
