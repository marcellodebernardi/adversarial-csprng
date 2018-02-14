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
This module provides miscellaneous utility functions for the training
and evaluation of the model, such as means for storing ASCII bit sequences
into text files and emailing a report after training.
"""

import sys
import numpy as np
from smtplib import SMTP
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from utils import operation_utils, input_utils
from keras import Model


def eprint(*args, **kwargs):
    """Prints to standard error."""
    # from https://stackoverflow.com/questions/5574702/how-to-print-to-stderr-in-python
    print(*args, file=sys.stderr, **kwargs)


def save_configuration(model, name):
    model.save('../output/saved_models/' + str(name) + '.h5', overwrite=True)


def generate_output_file(values, generator_name, max_value, val_bits):
    """Produces an ASCII output text file consisting of 0s and 1s.
    Such a file can be evaluated by the NIST test suite."""
    values = operation_utils.flatten_irregular_nested_iterable(values)
    binary_strings \
        = [('{:0>' + str(val_bits) + '}').format(bin(round(float(number))).replace('0b', '')) for number in values]

    with open('../output/sequences/' + str(generator_name) + '.txt', 'w') as file:
        for bin_str in binary_strings:
            file.write(str(bin_str) + "")


def log_adversary_predictions(gan: Model):
    inputs = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    outputs = gan.predict(inputs)
    print(outputs)

    inputs = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    outputs = gan.predict(inputs)
    print(outputs)

    with open('../output/logs/' + str(gan.name) + '.txt', 'w') as file:
        for pred in outputs:
            file.write(str(pred) + " ")


def email_report(batch_size, batches, unique_seeds, epochs, pretrain_epochs) -> bool:
    # sender and receiver
    sender = "neural.csprng@gmail.com"
    recipient = "marcello1234@live.co.uk"
    # create message
    msg = MIMEMultipart()
    # headers
    msg['neural.csprng@gmail.com'] = sender
    msg['marcello1234@live.co.uk'] = recipient
    msg['Subject'] = 'Adversarial CSPRNG Training Results'
    # body
    body = "Training complete. See attached files for model graphs, model state snapshots, and output sequences for" \
           + "evaluation.\n\n SETTINGS:\n" \
           + "Batch size: " + str(batch_size) + "\n" \
           + "Dataset size: " + str(batch_size * batches) + "\n" \
           + "Unique seeds per batch: " + str(unique_seeds) + "\n" \
           + "Training epochs: " + str(epochs) + "\n" \
           + "Pretraining epochs: " + str(pretrain_epochs) + "\n"
    msg.attach(MIMEText(body, 'plain'))
    # attachment list includes:
    # model graphs
    # saved models for reinstantiation
    # output files for NIST evaluation
    attachments = [
        ('jerry.h5', '../output/saved_models/'),
        ('diego.h5', '../output/saved_models/'),
        ('discgan.h5', '../output/saved_models/'),
        ('janice.h5', '../output/saved_models/'),
        ('priya.h5', '../output/saved_models/'),
        # ('predgan.h5', '../output/saved_models/'),
        ('jerry.png', '../output/model_graphs/'),
        ('diego.png', '../output/model_graphs/'),
        ('discriminative_gan.png', '../output/model_graphs/'),
        ('janice.png', '../output/model_graphs/'),
        ('priya.png', '../output/model_graphs/'),
        ('predictive_gan.png', '../output/model_graphs/'),
        ('diego_pretrain_loss.pdf', '../output/plots/'),
        ('discgan_train_loss.pdf', '../output/plots/'),
        ('priya_pretrain_loss.pdf', '../output/plots/'),
        ('predgan_train_loss.pdf', '../output/plots/'),
        ('discgan_jerry_output_distribution.pdf', '../output/plots/'),
        ('predgan_janice_output_distribution.pdf', '../output/plots/'),
        ('janice.txt', '../output/sequences/'),
        ('jerry.txt', '../output/sequences/')
    ]
    # insert attachments
    for att in attachments:
        filename = att[0]
        attachment = open(att[1] + att[0], 'rb')
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        msg.attach(part)

    server = SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender, 'neural_networks_rule_forever')
    text = msg.as_string()
    server.sendmail(sender, recipient, text)
    server.quit()
    return True
