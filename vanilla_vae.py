"""https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py"""
from typing import List, Callable

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Input


def encoder_block(
    x,
    filters: int,
    name: str = 'encoder_block'
):
    prefix = name
    x = layers.Conv2D(filters, 3, 2, 'same', name=f'{prefix}/conv')(x)
    x = layers.BatchNormalization(name=f'{prefix}/bn')(x)
    x = layers.Activation('leaky_relu', name=f'{prefix}/leaky_relu')(x)
    return x


def decoder_block(
    x,
    filters: int,
    name: str = 'decoder_block'
):
    prefix = name
    x = layers.Conv2DTranspose(
        filters,
        kernel_size=3,
        strides=2,
        padding='same',
        output_padding=1,
        name=f'{prefix}/deconv'
    )(x)
    x = layers.BatchNormalization(name=f'{prefix}/bn')(x)
    x = layers.Activation('leaky_relu', name=f'{prefix}/leaky_relu')(x)
    return x


def Encoder(
    x: tf.Tensor,
    filters_list: List[int],
    last_filters: int
):
    x = encoder_block(x, filters_list[0], name='encoder/block1')   # x/2
    x = encoder_block(x, filters_list[1], name='encoder/block2')   # x/4
    x = encoder_block(x, filters_list[2], name='encoder/block3')  # x/8
    x = encoder_block(x, filters_list[3], name='encoder/block4')  # x/16
    x = encoder_block(x, filters_list[4], name='encoder/block5')  # x/32

    feat_size = x.shape[1]  # square size

    x = layers.Flatten(name='encoder/flatten')(x)

    mu = layers.Dense(last_filters, name='encoder/dense/mu')(x)
    var = layers.Dense(last_filters, name='encoder/dense/var')(x)
    return mu, var, feat_size


def Decoder(
    x: tf.Tensor,
    feat_size: int,
    filters_list: List[int],
):
    x = layers.Dense(
        filters_list[-1] * feat_size * feat_size,
        name='decoder/dense'
    )(x)
    x = layers.Reshape(
        [feat_size, feat_size, filters_list[-1]],
        name='decoder/reshape'
    )(x)
    x = decoder_block(x, filters_list[-2], name='decoder/block1')
    x = decoder_block(x, filters_list[-3], name='decoder/block2')
    x = decoder_block(x, filters_list[-4], name='decoder/block3')
    x = decoder_block(x, filters_list[-5], name='decoder/block4')
    x = decoder_block(x, filters_list[-5], name='decoder/block5')
    x = layers.Conv2D(3, 3, 1, 'same', name='final/conv')(x)
    x = layers.Activation('tanh', name='final/tanh')(x)
    return x


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


def sampling(
    decoder: Callable,
    n_samples: int,
    latent_dims: int
):
    z = tf.random.normal([n_samples, latent_dims])
    samples = decoder(z)
    return samples


def VanillaVAE(
    input_shape: List[int] = [64, 64, 3],
    filters_list: List[int] = [32, 64, 128, 256, 512],
    encoder_last_dim: int = 128,
    name: str = 'vanilla_vae'
):
    """basic VAE"""
    inputs = Input(input_shape)
    mu, logvar, feat_size = Encoder(
        inputs,
        filters_list,
        encoder_last_dim
    )
    z = reparameterize(mu, logvar)
    output = Decoder(
        z,
        feat_size,
        filters_list
    )
    return Model(inputs, output, name=name)
