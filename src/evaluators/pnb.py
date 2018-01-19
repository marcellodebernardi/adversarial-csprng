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

import math
from tqdm import tqdm


def evaluate(file_path: str):
    """Evaluates a text file containing 0 and 1 ASCII characters
    using the practical next bit test."""
    with open(file_path, 'r') as file:
        # sequence properties and starting layer for pattern tree
        sequence = file.read()
        seq_length = len(sequence)
        start_layer = int(math.floor(math.log2(seq_length) - 2))
        decision_threshold = 0.55
        onb = 0

        # for each layer starting at start_layer, slide window over
        # entire sequence and count occurrences of each length-n
        # pattern's children of length n + 1
        for layer_index in tqdm(range(start_layer, seq_length), 'Computing PNB'):
            window_size = layer_index
            patterns = [PatternNode(i, layer_index) for i in range(seq_length)]
            padded_sequence = sequence + sequence[:window_size]

            for bit_index in range(seq_length):
                next_bit = int(padded_sequence[bit_index + window_size: bit_index + window_size + 1], 2)
                if next_bit == 1:
                    patterns[bit_index].one += 1
                else:
                    patterns[bit_index].zero += 1

            # count ONBs
            for pattern in patterns:
                total_occurrences = pattern.zero + pattern.one

                # onb += 1 if pattern occurs frequently and children are unbalanced
                if total_occurrences >= 5:
                    zero_ratio = pattern.zero / total_occurrences
                    if zero_ratio > decision_threshold or 1 - zero_ratio > decision_threshold:
                        onb += 1

    # more onbs is bad todo compute P value
    return onb


class PatternNode:
    def __init__(self, val: int, len: int):
        """Represents a pattern in the pattern tree used for the
        PNB algorithm. The integer val represents the value represented
        by the binary sequence in the node, and the len represents
        the number of bits the sequence consists of. That is, the
        tuple (val, len) specifies a binary string. For example, the
        tuple (3, 6) specifies the binary string 000011."""
        self.pattern = (val, len)
        self.zero = 0
        self.one = 0
