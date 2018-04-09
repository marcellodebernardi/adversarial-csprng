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
import os
import numpy as np
from smtplib import SMTP
from tqdm import tqdm
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from utils import operations


def eprint(*args, **kwargs):
    """ Prints to standard error. """
    # from https://stackoverflow.com/questions/5574702/how-to-print-to-stderr-in-python
    print(*args, file=sys.stderr, **kwargs)


def generate_output_hex(values, filename=None):
    """
    Produces an ASCII output text file containing hexadecimal representations
    of each number produces by the generator.
    """
    if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

    values = operations.flatten(values)
    values = [hex(np.uint16(i)) for i in values]

    with open(filename, 'w+') as file:
        for hex_val in tqdm(values, 'Writing to file ... '):
            file.write(str(hex_val) + "\n")


def log_to_file(logs, filename: str):
    """ Writes the given data array to the given file. No prettifying. """
    if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

    with open(filename, 'w+') as file:
        file.write(str(operations.flatten(logs)))


def email_report(batch_size, batches, epochs, pretrain_epochs) -> bool:
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
           + "Training epochs: " + str(epochs) + "\n" \
           + "Pretraining epochs: " + str(pretrain_epochs) + "\n\n\n"
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
        ('predgan.h5', '../output/saved_models/'),
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
        try:
            filename = att[0]
            attachment = open(att[1] + att[0], 'rb')
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
            msg.attach(part)
        except FileNotFoundError:
            body = body + "File not found: " + att[0] + "\n"

    server = SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender, 'neural_networks_rule_forever')
    text = msg.as_string()
    server.sendmail(sender, recipient, text)
    server.quit()
    return True
