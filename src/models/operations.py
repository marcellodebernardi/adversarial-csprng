import tensorflow as tf


def drop_last_bit(original_size):
    """Returns a closure that operates on Tensors. It is
    used to connect the GAN's generator to the discriminator,
    as the discriminator receives the generator's output
    minus the last bit."""

    def layer(x: tf.Tensor):
        return tf.reshape(x, [tf.shape(x)[0], original_size - 1])
    return layer
