# Marcello De Bernardi, Queen Mary University of London
# =================================================================

from keras.models import Model


class Network:
    """Defines the common attributes and behavior of both the
    generator and discriminator neural nets."""

    def __init__(self):
        self.inputs = None
        self.operations = None
        self.model = None
        self.optimizer = None
        self.loss_function = None

    def with_optimizer(self, optimizer):
        """Sets a new optimizer for the network."""
        self.optimizer = optimizer
        return self

    def with_loss_function(self, loss_function):
        """Sets a loss function for the network."""
        self.loss_function = loss_function
        return self

    def with_inputs(self, input_tensor):
        self.inputs = input_tensor
        return self

    def add_layer(self, new_layer):
        """Adds a layer to the Keras model"""
        if self.operations is None and self.inputs is not None:
            self.operations = new_layer(self.inputs)
        else:
            self.operations = new_layer(self.operations)
        return self

    def compile(self):
        """Compiles the Keras model for the network."""
        self.model = Model(self.inputs, self.operations)
        self.model.compile(self.optimizer, self.loss_function)
        return self

    def trainable(self, trainable=True):
        """Sets the trainability of the underlying Keras model.
        By default calling the method with no arguments sets the
        model to be trainable."""
        self.model.trainable = trainable
        for layer in self.model.layers:
            layer.trainable = trainable
        return self

    def get_model(self) -> Model:
        """Returns the underlying Keras model."""
        return self.model

    def get_input_tensor(self):
        """Returns the Tensor representing the input to the Network
        model."""
        return self.inputs

    def summary(self):
        """Prints a summary of the Network model."""
        self.model.summary()
        return
