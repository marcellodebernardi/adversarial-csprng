import sys
from keras import Model
from smtplib import SMTP
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from utils import input_utils, operation_utils


def eprint(*args, **kwargs):
    """Prints to standard error."""
    # from https://stackoverflow.com/questions/5574702/how-to-print-to-stderr-in-python
    print(*args, file=sys.stderr, **kwargs)


def save_configurations(disc_gan, pred_gan):
    disc_gan.get_model()[0].save('../saved_models/jerry.h5', overwrite=True)
    pred_gan.get_model()[0].save('../saved_models/janice.h5', overwrite=True)
    disc_gan.get_model()[1].save('../saved_models/diego.h5', overwrite=True)
    pred_gan.get_model()[1].save('../saved_models/priya.h5', overwrite=True)
    disc_gan.get_model()[2].save('../saved_models/disc_gan.h5', overwrite=True)
    # pred_gan.get_model()[2].save('../saved_models/pred_gan.h5', overwrite=True)


def generate_output_file(generator: Model, max_value, val_bits):
    """Produces an ASCII output text file consisting of 0s and 1s.
    Such a file can be evaluated by the NIST test suite."""
    values = generator.predict(input_utils.get_random_sequence(1, max_value))
    values = operation_utils.flatten_irregular_nested_iterable(values)
    binary_strings \
        = [('{:0>' + str(val_bits) + '}').format(bin(round(float(number))).replace('0b', '')) for number in values]

    with open('../sequences/' + str(generator.name) + '.txt', 'w') as file:
        for bin_str in binary_strings:
            file.write(str(bin_str) + "")


def email_report(settings, data_params) -> bool:
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
           + str(settings) + "\n\n" \
           + "DATA PARAMETERS:\n" \
           + str(data_params) + "\n\n"
    msg.attach(MIMEText(body, 'plain'))
    # attachment list includes:
    # model graphs
    # saved models for reinstantiation
    # output files for NIST evaluation
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
                   ('pred_gan.png', '../model_graphs/'),
                   ('disc_sequence.txt', '../sequences/'),
                   ('pred_sequence.txt', '../sequences/')
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
