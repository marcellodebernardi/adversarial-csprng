# Marcello De Bernardi, Queen Mary University of London
# =================================================================

import tensorflow as tf


class Network:
    """Defines the common attributes and behavior of both the
    generator and discriminator neural nets."""

    def __init__(self, session=None):
        self.session = session
        self.optimizer = tf.train.AdamOptimizer
        self.loss_function = None

    def with_session(self, session):
        """Sets the TensorFlow session for the network."""
        self.session = session
        return self

    def with_optimizer(self, optimizer):
        """Sets a new optimizer for the network."""
        self.optimizer = optimizer
        return self

    def with_loss_function(self, loss_function):
        """Sets a loss function for the network."""
        self.loss_function = loss_function
        return self

    def model(self):
        pass

    def optimize(self):
        """Runs the optimizer on the network. Returns after
        printing a warning if session has not yet been set."""
        if self.session is None:
            print("Session is None")
            return
        self.session.run(self.optimizer)
        return self

    def evaluate(self):
        """Computes the loss function of the network."""
        # todo
        return
