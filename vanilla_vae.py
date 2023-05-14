from typing import List

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Input


def VanillaVAE(
    input_shape: List[int, int, int]
):
    pass


def Encoder(
    x: tf.Tensor
):
    pass


def Decoder(
    x: tf.Tensor
):
    pass


def reparameterize(
    mu: tf.Tensor,
    logvar: tf.Tensor
):
    """
    Reparameterization trick to sample from N(mu, var)
    from N(0,1).
    """
    std = tf.math.exp(0.5 * logvar)
    eps = tf.random.normal(tf.shape(std))
    return std * eps + mu


def sampling():
    pass
