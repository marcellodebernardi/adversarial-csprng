def modular_activation(activation_function, modulo):
    """Activation function that uses the given standard activation
    function and then applies a modulo operation to its output."""
    def mod_act(input_value):
        return activation_function(input_value) % modulo
    return mod_act
