import matplotlib.pyplot as plt
from models.metrics import Metrics


def plot_metrics(metrics: Metrics, data_range: int):
    """Draws visual plots of all available data using matplotlib."""

    # generative/discriminative loss over epochs
    f = plt.figure(1)
    plt.plot(metrics.generator_loss(), metrics.predictor_loss())
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlabel('Gradient Update Iteration')
    plt.legend()
    f.show()
    # distribution of generated values during training
    g = plt.figure(2)
    plt.hist(metrics.generator_outputs(), bins=data_range * 3)
    plt.title('Generator Output Distribution (Training)')
    plt.xlabel('Output')
    plt.ylabel('Frequency')
    g.show()
    # distribution of generated values during evaluation
    h = plt.figure(3)
    plt.hist(metrics.generator_eval_outputs(), bins=data_range * 3)
    plt.title('Generator Output Distribution (Evaluation)')
    plt.xlabel('Output')
    plt.ylabel('Frequency')
    h.show()
    # Average output per batch during training
    i = plt.figure(4)
    plt.plot(metrics.generator_avg_outputs())
    plt.ylabel('Average output per batch')
    plt.xlabel('Batch training iteration')
    i.show()

    plt.show()


def plot_model_weights(generator_avg_weights, predictor_avg_weights):
    plt.plot(generator_avg_weights)
    plt.ylabel('Average Generator Weight')
    plt.xlabel('Gradient Update Iteration')
