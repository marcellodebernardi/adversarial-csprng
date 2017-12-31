import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.metrics import Metrics


def plot_metrics(metrics: Metrics, data_range: int):
    """Plots all metrics """
    plot_loss(metrics.generator_loss(), metrics.predictor_loss())
    plot_generator_outputs(metrics.generator_outputs(), data_range)


def plot_loss(generator_loss, predictor_loss):
    """Plots the generator and predictor loss values into a line
    chart.
    """
    ax = pd.DataFrame(
        {
            'Generative Loss': generator_loss,
            'Predictive Loss': predictor_loss,
        }
    ).plot(title='Training loss')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    plt.show()


def plot_generator_outputs(outputs: np.ndarray, data_range):
    """Plots the output values of the generator into a histogram
    to visualize the output value distribution.
    """
    plt.hist(outputs, bins=data_range * 3)
    plt.title('Generator Output Distribution')
    plt.xlabel('Output')
    plt.ylabel('Frequency')
    plt.show()


def plot_model_weights(generator_avg_weights, predictor_avg_weights):
    plt.plot(generator_avg_weights)
    plt.ylabel('Average Generator Weight')
    plt.xlabel('Gradient Update Iteration')
    plt.show()
