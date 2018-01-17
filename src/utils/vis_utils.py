import matplotlib.pyplot as plt
from models.gan_predictive import PredictiveGan
from models.gan_discriminative import DiscriminativeGan
from models.metrics import Metrics
from keras import Model
from keras.utils import plot_model


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

    # distribution of weights in generator
    fig4 = plt.figure()
    plt.subplot(211)
    plt.hist(metrics.generator_weights_initial(), bins=300)
    plt.title('Initial Generator Weight Distribution')
    plt.ylabel('Frequency')
    plt.xlabel('Weight')
    plt.subplot(212)
    plt.hist(metrics.generator_weights_final(), bins=300)
    plt.title('Final Generator Weight Distribution')
    plt.ylabel('Frequency')
    plt.xlabel('Weight')
    fig4.show()

    # distribution of weights in predictor
    fig5 = plt.figure()
    plt.subplot(211)
    plt.hist(metrics.predictor_weights_initial(), bins=300)
    plt.title('Initial Predictor Weight Distribution')
    plt.ylabel('Frequency')
    plt.xlabel('Weight')
    plt.subplot(212)
    plt.hist(metrics.predictor_weights_final(), bins=300)
    plt.title('Final Predictor Weight Distribution')
    plt.ylabel('Frequency')
    plt.xlabel('Weight')
    fig5.show()

    plt.show()


def plot_network_graphs(gan_model: (Model, Model, Model), prefix: str):
    """Draws visualizations of the network structure as well as the
    shape of each layer."""
    plot_model(gan_model[0], '../model_graphs/' + prefix + '_generator.png', show_shapes=True)
    plot_model(gan_model[1], '../model_graphs/' + prefix + '_adversary.png', show_shapes=True)
    plot_model(gan_model[2], '../model_graphs/' + prefix + '_gan.png', show_shapes=True)
