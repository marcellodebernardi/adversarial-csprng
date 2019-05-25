# Marcello De Bernardi, University of Oxford
#
# An exploratory proof-of-concept implementation of a CSPRNG
# (cryptographically secure pseudorandom number generator) using
# adversarially trained neural networks. The work was inspired by
# the findings of Abadi & Andersen, outlined in the paper
# "Learning to Protect Communications with Adversarial
# Networks", available at https://arxiv.org/abs/1610.06918.
#
# The original implementation by Abadi is available at
# https://github.com/tensorflow/models/tree/master/research/adversarial_crypto
# =================================================================

import math
from abc import ABC, abstractmethod
from components import inputs

# main settings
NUMBER_OF_BITS_PRODUCED = 1000000000
# hyper-parameter with list of options for robustness testing
V = {
    'input_size': {'default': 2, 'options': [1, 2, 3, 4, 5]},
    'gen_width': {'default': 32, 'options': [8, 16, 32, 64, 128, 256, 512, 1024]},
    'output_size': {'default': 8, 'options': [8, 16, 32, 64, 128, 256, 512, 1024]},
    'output_bits': {'default': 16, 'options': [2, 4, 8, 16, 32]},
    'max_val': {'default': 65535, 'options': [3, 15, 255, 65535, 4294967295]},
    'batch_size': {'default': 2048, 'options': [8, 16, 32, 64, 128, 256, 512, 1024, 2048]},
    'learning_rate': {'default': 0.010, 'options': [0.001, 0.010, 0.020, 0.040, 0.080, 0.16, 0.32, 0.64]},
    'adv_mul': {'default': 3, 'options': [1, 2, 3, 4, 5]}
}
EVAL_SEED = 10


class ExperimentFunction(ABC):
    @abstractmethod
    def run_function(self, input_size, gen_width, output_size, max_val, batch_size, learning_rate, adv_mul, steps,
                     eval_data, folder):
        pass


class Experiment:
    def __init__(self, experiment_function: ExperimentFunction, repetitions: int, independent_variable_name=None):
        if independent_variable_name not in list(V.keys()):
            raise ValueError('Independent variable must be one of ' + str(list(V.keys())))
        self.experiment_function = experiment_function.run_function
        self.repetitions = repetitions
        self.independent_variable_name = independent_variable_name

    def perform(self, steps):
        # baseline experiment with default parameters
        if self.independent_variable_name is None:
            eval_data = __generate_eval_dataset__()
            self.experiment_function.run_function(
                V['input_size']['default'],
                V['gen_width']['default'],
                V['output_size']['default'],
                V['max_val']['default'],
                V['batch_size']['default'],
                V['learning_rate']['default'],
                V['adv_mul']['default'],
                steps,
                eval_data,
                'defaults/')
        # if independent variable is input size, which affects eval dataset structure
        elif self.independent_variable_name == 'input_size':
            for input_size in V['input_size']['options']:
                eval_data = __generate_eval_dataset__(input_size=input_size)
                for rep in range(self.repetitions):
                    self.experiment_function.run_function(
                        input_size,
                        V['gen_width']['default'],
                        V['output_size']['default'],
                        V['max_val']['default'],
                        V['batch_size']['default'],
                        V['learning_rate']['default'],
                        V['adv_mul']['default'],
                        steps,
                        eval_data,
                        'input_size/' + str(input_size) + '/')
        # if independent variable is gen width size
        elif self.independent_variable_name == 'gen_width':
            eval_data = __generate_eval_dataset__()
            for gen_width in V['gen_width']['options']:
                for rep in range(self.repetitions):
                    self.experiment_function.run_function(
                        V['input_size']['default'],
                        gen_width,
                        V['output_size']['default'],
                        V['max_val']['default'],
                        V['batch_size']['default'],
                        V['learning_rate']['default'],
                        V['adv_mul']['default'],
                        steps,
                        eval_data,
                        'gen_width/' + str(gen_width) + '/')
        # if independent variable is output size, which affects eval dataset structure
        elif self.independent_variable_name == 'output_size':
            for output_size in V['output_size']['options']:
                eval_data = __generate_eval_dataset__(output_size=output_size)
                for rep in range(self.repetitions):
                    self.experiment_function.run_function(
                        V['input_size']['default'],
                        V['gen_width']['default'],
                        output_size,
                        V['max_val']['default'],
                        V['batch_size']['default'],
                        V['learning_rate']['default'],
                        V['adv_mul']['default'],
                        steps,
                        eval_data,
                        'output_size/' + str(output_size) + '/')
        # if independent variable is max_value, which affects eval dataset structure
        elif self.independent_variable_name == 'max_val':
            for max_val in V['max_val']['options']:
                eval_data = __generate_eval_dataset__(output_bits=round(math.log(max_val, 2)))
                for rep in range(self.repetitions):
                    self.experiment_function.run_function(
                        V['input_size']['default'],
                        V['gen_width']['default'],
                        V['output_size']['default'],
                        max_val,
                        V['batch_size']['default'],
                        V['learning_rate']['default'],
                        V['adv_mul']['default'],
                        steps,
                        eval_data,
                        'max_val/' + str(max_val) + '/')
        # if independent variable is batch_size, which does not affect dataset
        elif self.independent_variable_name == 'batch_size':
            eval_data = __generate_eval_dataset__()
            for batch_size in V['batch_size']['options']:
                for rep in range(self.repetitions):
                    self.experiment_function.run_function(
                        V['input_size']['default'],
                        V['gen_width']['default'],
                        V['output_size']['default'],
                        V['max_val']['default'],
                        batch_size,
                        V['learning_rate']['default'],
                        V['adv_mul']['default'],
                        steps,
                        eval_data,
                        'batch_size/' + str(batch_size) + '/')
        elif self.independent_variable_name == 'learning_rate':
            eval_data = __generate_eval_dataset__()
            for learning_rate in V['learning_rate']['options']:
                for rep in range(self.repetitions):
                    self.experiment_function.run_function(
                        V['input_size']['default'],
                        V['gen_width']['default'],
                        V['output_size']['default'],
                        V['max_val']['default'],
                        V['batch_size']['default'],
                        learning_rate,
                        V['adv_mul']['default'],
                        steps,
                        eval_data,
                        'learning_rate/' + str(learning_rate) + '/')


def __generate_eval_dataset__(input_size=V['input_size']['default'], output_size=V['output_size']['default'],
                              output_bits=V['output_bits']['default']):
    num_of_elements = NUMBER_OF_BITS_PRODUCED / (output_size * output_bits)
    return inputs.get_eval_input_numpy(EVAL_SEED, num_of_elements, input_size)
