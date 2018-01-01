import matplotlib.pyplot as plt
from models.metrics import Metrics


def plot_metrics(metrics: Metrics, data_range: int):
    """Draws visual plots of all available data using matplotlib."""

    # output distribution plot
    fig1 = plt.figure()
    # distribution of generated values during training
    plt.subplot(211)
    plt.hist(metrics.generator_outputs(), bins=data_range * 3)
    plt.title('Generator Output Distribution (Training)')
    plt.xlabel('Output')
    plt.ylabel('Frequency')
    # distribution of generated values during evaluation
    plt.subplot(212)
    plt.hist(metrics.generator_eval_outputs(), bins=data_range * 3)
    plt.title('Generator Output Distribution (Evaluation)')
    plt.xlabel('Output')
    plt.ylabel('Frequency')
    fig1.show()

    # loss value charts
    fig2 = plt.figure()
    plt.plot(metrics.generator_loss())
    plt.plot(metrics.predictor_loss())
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Generative loss', 'Predictive loss'])
    fig2.show()

    # Average output per batch during training
    fig3 = plt.figure()
    plt.plot(metrics.generator_avg_outputs())
    plt.ylabel('Average output per batch')
    plt.xlabel('Batch training iteration')
    fig3.show()

    # distribution of weights in networks
    fig4 = plt.figure()
    plt.hist()

    plt.show()


def plot_model_weights(generator_avg_weights, predictor_avg_weights):
    plt.plot(generator_avg_weights)
    plt.ylabel('Average Generator Weight')
    plt.xlabel('Gradient Update Iteration')
