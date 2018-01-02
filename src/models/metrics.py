import numpy as np

class Metrics:
    """A data storage class for the various metrics that
    may be collected during training.
    """

    def __init__(self):
        self.__generator_loss = []
        self.__predictor_loss = []
        self.__predictor_pretrain_loss = []
        self.__generator_outputs = []
        self.__generator_avg_outputs = []
        self.__generator_eval_outputs = []
        self.__generator_weights_initial = []
        self.__predictor_weights_initial = []
        self.__generator_weights_final = []
        self.__predictor_weights_final = []

    def generator_loss(self) -> list:
        """Returns reference to the list that is supposed to hold
        the time series of the generator's loss function.
        """
        return self.__generator_loss

    def predictor_loss(self) -> list:
        """Returns reference to the list that is supposed to hold
        the time series of the predictor's loss function.
        """
        return self.__predictor_loss

    def predictor_pretrain_loss(self) -> list:
        """Returns reference to the list that is supposed to hold
        the time series of the predictor's loss function for the
        pre-training period."""
        return self.__predictor_pretrain_loss

    def generator_outputs(self) -> list:
        """Returns reference to the list that is supposed to hold
        the collection of all values outputted by the generator
        during training.
        """
        return self.__generator_outputs

    def generator_eval_outputs(self) -> list:
        """Returns reference to the list that is supposed to hold
        the collection of all values outputted by the generator
        during evaluation.
        """
        return self.__generator_eval_outputs

    def generator_avg_outputs(self) -> list:
        """Returns reference to the list that is supposed to hold
        the time series of the average output value of the generator
        at each epoch of training.
        """
        return self.__generator_avg_outputs

    def generator_weights_initial(self) -> list:
        """Returns reference to the list that is supposed to hold
        the collection of initial generator weights.
        """
        return self.__generator_weights_initial

    def generator_weights_final(self) -> list:
        """Returns reference to the list that is supposed to hold
        the collection of final generator weights.
        """
        return self.__generator_weights_final

    def predictor_weights_initial(self) -> list:
        """Returns reference to the list that is supposed to hold
        the collection of initial predictor weights.
        """
        return self.__predictor_weights_initial

    def predictor_weights_final(self) -> list:
        """Returns reference to the list that is supposed to hold
        the collection of final predictor weights.
        """
        return self.__predictor_weights_final
