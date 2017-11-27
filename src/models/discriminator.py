# Marcello De Bernardi, Queen Mary University of London
# =================================================================
import tensorflow as tf
from models.network import Network


class Discriminator(Network):
    """Defines the discriminator neural network model, 'Eve'."""

    def __init__(self, use_oracle):
        super().__init__()
        self.use_oracle = use_oracle
