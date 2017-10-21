import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main():
    # import data
    mnist = input_data.read_data_sets(one_hot=True)

    # create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # define loss and optimizer
    labels = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss=False)