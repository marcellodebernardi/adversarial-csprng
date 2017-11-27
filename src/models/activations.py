def modular_activation(modulo):
    def mod_act(input_value):
        return input_value % modulo
    return mod_act
