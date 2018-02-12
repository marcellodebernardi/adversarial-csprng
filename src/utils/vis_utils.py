import matplotlib.pyplot as plt
from keras import Model
from keras.utils import plot_model
from utils.operation_utils import flatten_irregular_nested_iterable


def plot_pretrain_history_loss(history, fname):
    """Plot a line chart of the adversary's loss during pre-training."""
    plt.plot(history.history['loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(fname)
    plt.clf()


def plot_train_loss(generator_loss, adversary_loss, fname):
    """Plot a line chart of the generator's and adversary's losses
    during training. """
    plt.plot(generator_loss)
    plt.plot(adversary_loss)
    plt.ylabel('Loss')
    plt.ylabel('Epoch')
    plt.legend(['Generative loss', 'Adversary loss'])
    plt.savefig(fname)
    plt.clf()


def plot_output_histogram(values, fname):
    """Plot a histogram of the output values for one seed. """
    values = flatten_irregular_nested_iterable(values)
    plt.hist(values, bins=int((max(values) - min(values))*3))
    plt.title('Generator Output Frequency Distribution')
    plt.xlabel('Output')
    plt.ylabel('Frequency')
    plt.savefig(fname)
    plt.clf()


def plot_output_sequence(values, fname):
    """Plot a line displaying the sequence of output values
    for a trained generator, for one seed, in temporal order."""
    plt.plot(values)
    plt.ylabel('Output')
    plt.xlabel('Position in Sequence')
    plt.savefig(fname)
    plt.clf()


def plot_network_weights(weights, fname):
    """Plots a histogram of network weights."""
    plt.hist(weights, bins=int((max(weights) - min(weights))*100))
    plt.ylabel('Frequency')
    plt.xlabel('Weight')
    plt.savefig(fname)
    plt.clf()


def plot_metrics(metrics, data_range: int):
    """Draws visual plots of all available data using matplotlib."""
    # output distribution plot
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


def plot_network_graphs(model: Model, name: str):
    """Draws visualizations of the network structure as well as the
    shape of each layer."""
    plot_model(model, '../output/model_graphs/' + name + '.png', show_shapes=True)
