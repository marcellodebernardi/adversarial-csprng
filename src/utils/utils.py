import sys
import tensorflow as tf
import numpy as np
import datetime
from keras import Model
from smtplib import SMTP
from email.mime.multipart import MIMEMultipart
from email.mime.text import  MIMEText
from email.mime.base import MIMEBase
from email import encoders


def split_generator_output(generator_output: np.ndarray, n_to_predict) -> (np.ndarray, np.ndarray):
    """Takes the generator output as a numpy array and splits it into two
    separate numpy arrays, the first representing the input to the predictor
    and the second representing the output labels for the predictor."""
    batch_len = len(generator_output)
    seq_len = len(generator_output[0])
    predictor_inputs = generator_output[0: batch_len, 0: -n_to_predict]
    predictor_outputs = generator_output[0: batch_len, seq_len - n_to_predict - 1: seq_len - n_to_predict]
    return predictor_inputs, predictor_outputs


def set_trainable(model: Model, trainable: bool = True):
    """Helper method that sets the trainability of all of a model's
    parameters."""
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def log(x, base) -> tf.Tensor:
    """Allows computing element-wise logarithms on a Tensor, in
    any base. TensorFlow itself only has a natural logarithm
    operation."""
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator


def flatten_irregular_nested_iterable(weight_matrix) -> list:
    """Allows flattening a matrix of iterables where the specific type
    and shape of each iterable is not necessarily the same. Returns
    the individual elements of the original nested iterable in a single
    flat list.
    """
    flattened_list = []
    try:
        for element in weight_matrix:
            flattened_list.extend(flatten_irregular_nested_iterable(element))
        return flattened_list
    except TypeError:
        return weight_matrix


def eprint(*args, **kwargs):
    """Prints to standard error."""
    # from https://stackoverflow.com/questions/5574702/how-to-print-to-stderr-in-python
    print(*args, file=sys.stderr, **kwargs)


def save_configurations(disc_gan, pred_gan):
    disc_gan.get_model()[0].save('../saved_models/disc_generator.h5', overwrite=True)
    pred_gan.get_model()[0].save('../saved_models/pred_generator.h5', overwrite=True)
    disc_gan.get_model()[1].save('../saved_models/disc_adversary.h5', overwrite=True)
    pred_gan.get_model()[1].save('../saved_models/pred_adversary.h5', overwrite=True)
    disc_gan.get_model()[2].save('../saved_models/disc_gan.h5', overwrite=True)
    # pred_gan.get_model()[2].save('../saved_models/pred_gan.h5', overwrite=True)


def email_report() -> bool:
    # sender and receiver
    fromaddr = "neural.csprng@gmail.com"
    toaddr = "marcello1234@live.co.uk"
    # create message
    msg = MIMEMultipart()
    # headers
    msg['neural.csprng@gmail.com'] = fromaddr
    msg['marcello1234@live.co.uk'] = toaddr
    msg['Subject'] = 'Adversarial CSPRNG Training Results'
    # body
    body = "Training complete"
    msg.attach(MIMEText(body, 'plain'))
    # attachment list
    attachments = [
                   ('disc_generator.h5', '../saved_models/'),
                   ('disc_adversary.h5', '../saved_models/'),
                   ('disc_gan.h5', '../saved_models/'),
                   ('pred_generator.h5', '../saved_models/'),
                   ('pred_adversary.h5', '../saved_models/'),
                   ('pred_gan.h5', '../saved_models/'),
                   ('disc_generator.png', '../model_graphs/'),
                   ('disc_adversary.png', '../model_graphs/'),
                   ('disc_gan.png', '../model_graphs/'),
                   ('pred_generator.png', '../model_graphs/'),
                   ('pred_adversary.png', '../model_graphs/'),
                   ('pred_gan.png', '../model_graphs/')]
    # insert attachments
    for att in attachments:
        filename = att[0]
        attachment = open(att[1] + att[0])
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        msg.attach(part)
    server = SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, 'neural_networks_rule_forever')
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
    return True
