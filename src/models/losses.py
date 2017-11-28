from keras import backend as K
import math
import numpy as np


def loss_nist(true, pred):
    """Computes the loss function for an output using NIST"""
    pass


def loss_ent(true, pred):
    """Computes the loss function for an output using ent"""
    pass


def loss_pnb(true, pred):
    """Computes the loss function for an output using the
    practical next bit test"""
    # todo change implementation to work with Tensors
    output = K.eval(pred).ndarray.tolist()
    output_length = len(output)
    decision_threshold = 0.55  # todo pick sensible value
    onb = 0

    # start from patterns of length
    start_length = math.log(len(output), 2) - 2

    # add every possible pattern of length start_length to
    # a list; the pattern is not explicitly stored, rather
    # the binary representation of the list index of each
    # node is the corresponding pattern
    node_list = []
    for i in range(int(2 ** start_length)):
        node_list.append(Node())

    # append first start_length values to end of list
    for i in range(int(start_length)):
        output.append(output[i])

    # slide window over list and count pattern observations
    for i in range(output_length):
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


class Node:
    def __init__(self):
        self.zero = 0
        self.one = 0
