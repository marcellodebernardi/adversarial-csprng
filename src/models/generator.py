# Marcello De Bernardi, Queen Mary University of London
# =================================================================
import tensorflow as tf
import os
from models.network import Network


class Generator(Network):
    """Defines the generator neural network model, 'Jerry'. """

    def __init__(self):
        super().__init__()
