# Marcello De Bernardi, Queen Mary University of London
# =================================================================
import tensorflow as tf
import os

class Generator:
    """Defines the generator neural network model, 'Jerry'. """

    def __init__(self, withOracle = False, seed_length):
        self.seed_length = seed_length
        self.seed = os.urandom(self.seed_length/8)
        return self
    

    def evaluate(self, session, n):
        """Computes and returns the value of the loss function of the
        generator network."""


    def reseed(self):
        """Reseeds the generator"""
        self.seed = os.urandom(self.seed_length/8)
