import matplotlib.pyplot as plt
from keras import Model
from keras.utils import plot_model


def plot_pretrain_history_loss(history, fname):
    print(history.history['loss'])
    plt.plot(history.history['loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(fname)
    plt.clf()


def plot_train_loss(generator_loss, adversary_loss, fname):
    plt.plot(generator_loss)
    plt.plot(adversary_loss)
    plt.ylabel('Loss')
    plt.ylabel('Epoch')
    plt.legend(['Generative loss', 'Adversary loss'])
    plt.savefig(fname)
    plt.clf()


def plot_metrics(metrics, data_range: int):
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
    # pretrain loss
    fig6 = plt.figure()
    plt.plot(metrics.adversary_pretrain_loss())
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    fig6.show()
    # show all plots
    plt.show()


def plot_network_graphs(model: Model, name: str):
    """Draws visualizations of the network structure as well as the
    shape of each layer."""
    plot_model(model, '..output/model_graphs/' + name + '.png', show_shapes=True)
