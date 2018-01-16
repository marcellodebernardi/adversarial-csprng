# Marcello De Bernardi, Queen Mary University of London
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
"""
This module provides facilities for evaluating the output of a
PRNG using the 'practical next bit test'.
"""


def loss_pnb(output_length: int):
    """Returns a loss compatible with the Keras specification, that
    computes the loss of the generator based on the practical next
    bit test."""

    def loss_pnb_closure(true, pred):
        """Loss function that computes the practical next bit test on
        an output tensor representing the sequence produced by the
        generator"""
        # NOTE: the loss function must be purely symbolic and operate on
        # tensors.

        print(tf.shape(pred))

        data = tf.reshape(tf.cast(pred, dtype=tf.float32), [output_length])
        onb = tf.Variable([0], dtype=tf.float32)

        # compute length of patterns as log_2(output_size) - 2
        start_length = tf.subtract(
            utils.log(tf.Variable(output_length, dtype=tf.float32), 2),
            tf.constant(2, dtype=data.dtype))

        # todo append first start_length values to end of list
        # todo tf.concat([data, tf.slice(data, tf.constant(0), tf.cast(start_length, tf.int32))], 0)

        pattern_list = tf.zeros([output_length, 2], tf.float32)

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