class Metrics:
    """A data storage class for the various metrics that
    may be collected during training.
    """

    def __init__(self):
        self.__generator_loss = []
        self.__predictor_loss = []
        self.__generator_outputs = []
        self.__generator_final_outputs = []
        self.__generator_max_weight = []
        self.__generator_min_weight = []
        self.__generator_avg_weight = []
        self.__predictor_max_weight = []
        self.__predictor_min_weight = []
        self.__predictor_avg_weight = []
        self.__generator_final_weights = []
        self.__predictor_final_weights = []

    def generator_loss(self) -> list:
        return self.__generator_loss

    def predictor_loss(self) -> list:
        return self.__predictor_loss

    def generator_outputs(self) -> list:
        return self.__generator_outputs

    def generator_final_outputs(self) -> list:
        return self.__generator_final_outputs
