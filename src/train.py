import utils.utils as utils
import utils.nn_utils as nn_utils
import numpy as np
from keras import Model
from tqdm import tqdm
from models.metrics import Metrics


def evaluate(generator: Model, adversarial: Model, seed_dataset, metrics: Metrics):
    """Performs evaluation of the generator/adversarial model and updates
    the metrics object with the evaluation results.
    """
    for generator_input in tqdm(seed_dataset, desc='Eval: '):
        generator_output = generator.predict_on_batch(generator_input)
        metrics.generator_eval_outputs().extend(generator_output.flatten())
