import utils
import tensorflow as tf


def loss_nist(true, pred):
    """Computes the loss function for an output using NIST"""
    pass


def loss_pnb(output_size):
    def loss_pnb_closure(true, pred):
        """Loss function that computes the practical next bit test on
        an output tensor representing the sequence produced by the
        generator"""
        # NOTE: the loss function must be purely symbolic and operate on
        # tensors.

        data = tf.reshape(tf.cast(pred, dtype=tf.float32), [output_size])
        onb = tf.Variable([0], dtype=tf.float32)

        # compute length of patterns as log_2(output_size) - 2
        start_length = tf.subtract(
            utils.log(tf.Variable(output_size, dtype=tf.float32), 2),
            tf.constant(2, dtype=data.dtype))

        # todo append first start_length values to end of list
        # todo tf.concat([data, tf.slice(data, tf.constant(0), tf.cast(start_length, tf.int32))], 0)

        pattern_list = tf.zeros([output_size, 2], tf.float32)

        # todo

        return data

    return loss_pnb_closure


def loss_ent(true, pred):
    """Computes the loss function for an output using ent"""

    def loss_pnb(output_size, decision_threshold=0.55):
        def loss_pnb_closure(true, pred):
            """Computes the loss function for an output using the
            practical next bit test"""

            # slide window over list and count pattern observations
            for i in range(output_size):
                pattern = ''
                for j in range(i, int(i + start_length)):
                    pattern += output[j]
                pattern = int(pattern, 2)

                if output[i + start_length + 1] is 0:
                    node_list[pattern].zero += 1
                else:
                    node_list[pattern].one += 1

            # count ONBs
            for i in range(len(node_list)):
                total_occurrences = node_list[i].zero + node_list[i].one

                if total_occurrences >= 5:
                    zero_ratio = node_list[i].zero / total_occurrences

                    if zero_ratio > decision_threshold or 1 - zero_ratio > decision_threshold:
                        onb += 1

            # more onbs is bad todo compute P value
            return onb

        return loss_pnb_closure

    pass


def loss_disc(true, pred):
    """Loss function for the discriminator network."""
    pass


def loss_gan(true, pred):
    """Loss function for the GAN training phase. Precisely this is
    the loss function for the generator, but the Keras model trains
    the generator as a part of the larger GAN model."""
    return tf.subtract(true, pred)


class Node:
    def __init__(self):
        self.zero = 0
        self.one = 0
