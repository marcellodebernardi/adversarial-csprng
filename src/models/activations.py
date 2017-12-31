def modulo(divisor, activation_function=None):
    """Activation function that uses the given standard activation
        function and then applies a modulo operation to its output."""
    def mod_act(input_value):
        if activation_function is not None:
            return activation_function(input_value) % divisor
        else:
            return input_value % divisor
    return mod_act


def absolute(input_value):
    return abs(input_value)
