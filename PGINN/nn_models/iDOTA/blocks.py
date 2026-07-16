# -*- coding: utf-8 -*-
# Building blocks of the data-driven dose calculator.
# COPYRIGHT: TU Delft, Netherlands. 2021.
import numpy as np
import tensorflow as tf

# from tensorflow.keras.regularizers import Regularizer
# from tensorflow.keras import Sequential, layers
# from tensorflow.keras.layers import TimeDistributed as td

from keras import Sequential, layers
from keras.layers import TimeDistributed as td
from keras.regularizers import Regularizer

class CatLayer(layers.Layer):
    """Concatenate a list of tensors along the last axis."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, training=None, mask=None):
        return tf.concat(x, axis=-1)

    def get_config(self):
        config = super().get_config()
        return config

class ConvEncoder(layers.Layer):
    """Tokenize 2D using a series of convolutional layers."""
    def __init__(self, filters, steps=4, num_channels=32, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.projection_channels = filters
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.down_steps = steps

        # placeholders for sublayers
        self.encoder = None

    def build(self, input_shape):
        self.encoder = Sequential(name="conv_encoder_seq")
        for _ in range(self.down_steps):
            self.encoder.add(ConvBlock(self.num_channels,
                                       self.kernel_size,
                                       downsample=True))
        self.encoder.add(ConvBlock(self.projection_channels,
                                   self.kernel_size,
                                   flatten=True))
        super().build(input_shape)  # mark as built

    def call(self, volumes, training=None, mask=None):
        return self.encoder(volumes)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.projection_channels,
            "steps": self.down_steps,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ConvDecoder(layers.Layer):
    """Convert transformer output to dose."""
    def __init__(self, token_dim, steps=4, num_channels=32, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.token_dim = token_dim
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.up_steps = steps

        # placeholders for sublayers
        self.reshape = None
        self.decoder = None

    def build(self, input_shape):
        # Reshape to 3D
        self.reshape = td(layers.Reshape((self.token_dim,)))

        # Convolutional transpose layers
        self.decoder = Sequential(name="conv_decoder_seq")
        for _ in range(self.up_steps):
            self.decoder.add(ConvBlock(self.num_channels, self.kernel_size, upsample=True))
        self.decoder.add(td(layers.Conv2D(1, self.kernel_size, padding='same')))

        super().build(input_shape)  # mark as built

    def call(self, tokens, training=None, mask=None):
        return self.decoder(self.reshape(tokens))

    def get_config(self):
        config = super().get_config()
        config.update({
            "token_dim": self.token_dim,
            "steps": self.up_steps,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class TransformerEncoder(layers.Layer):
    """Transformer encoder block."""
    def __init__(self, num_heads, num_tokens, projection_dim,
                 causal=True, dropout_rate=0.2, num_forward=0, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.projection_dim = int(np.prod(projection_dim))
        self.causal = causal
        self.dropout_rate = dropout_rate
        self.num_forward = num_forward

        # placeholders for sublayers
        self.multihead = None
        self.mlp_network = None
        self.norm1 = None
        self.norm2 = None
        self.add = None
        self.mask = None

    def build(self, input_shape):
        # Multi-head self attention
        self.multihead = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.projection_dim,
            dropout=self.dropout_rate,
            kernel_initializer='truncated_normal',
            use_bias=False
        )

        # MLP stack
        self.mlp_network = Sequential([
            layers.Dense(self.projection_dim, activation=tf.nn.gelu),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.projection_dim, activation=tf.nn.gelu),
            layers.Dropout(self.dropout_rate)
        ])

        # Normalization and residual add
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.add = layers.Add()

        # Mask for causal attention
        if self.causal:
            self.mask = np.tri(self.num_tokens, self.num_tokens,
                               self.num_forward, dtype=bool)
        else:
            self.mask = np.ones((self.num_tokens, self.num_tokens))

        super().build(input_shape)  # mark as built

    def call(self, tokens, training=None, mask=None):
        x = self.norm1(tokens)
        x = self.multihead(x, x, attention_mask=self.mask)
        x = self.add([x, tokens])
        y = self.norm2(x)
        y = self.mlp_network(y)
        return self.add([x, y])

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "num_tokens": self.num_tokens,
            "projection_dim": self.projection_dim,
            "causal": self.causal,
            "dropout_rate": self.dropout_rate,
            "num_forward": self.num_forward,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


##################################
# Auxiliary blocks and functions
##################################

class ConvBlock(layers.Layer):
    """Down-sampling / up-sampling convolutional block."""
    def __init__(self, num_channels=64, kernel_size=3,
                 downsample=False, upsample=False, flatten=False, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.upsample = upsample
        self.flatten = flatten

        # placeholders for sublayers
        self.block = None

    def build(self, input_shape):
        self.block = Sequential(name="conv_block_seq")
        self.block.add(layers.Conv3D(
            self.num_channels, self.kernel_size,
            use_bias=False,
            padding='same'
        ))
        if self.downsample:
            self.block.add(layers.MaxPooling3D(pool_size=(1, 2, 2)))
        if self.upsample:
            self.block.add(layers.UpSampling3D(size=(1, 2, 2)))
        self.block.add(layers.LayerNormalization())
        self.block.add(layers.LeakyReLU())
        if self.flatten:
            self.block.add(td(layers.Flatten('channels_last')))
        super().build(input_shape)  # mark as built

    def call(self, inputs, training=None, mask=None):
        return self.block(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "downsample": self.downsample,
            "upsample": self.upsample,
            "flatten": self.flatten,
        })
        return config


class PosEmbedding(layers.Layer):
    """Add positional embeddings to a sequence of tokens."""
    def __init__(self, num_tokens, token_size, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.token_size = token_size
        self.embedding = None  # will be created in build()

    def build(self, input_shape):
        # Create the embedding layer once input shape is known
        self.embedding = layers.Embedding(
            input_dim=self.num_tokens,
            output_dim=self.token_size
        )
        super().build(input_shape)  # mark as built

    def call(self, inputs, training=None, mask=None):
        # inputs: (batch, seq_len, token_size)
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_emb = self.embedding(positions)  # (seq_len, token_size)
        return inputs + pos_emb

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_tokens": self.num_tokens,
            "token_size": self.token_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class LinearProj(layers.Layer):
    """Project scalars to token vectors."""
    def __init__(self, token_size, **kwargs):
        super().__init__(**kwargs)
        self.token_size = token_size
        self.projection = None
        self.flatten = None

    def build(self, input_shape):
        # Create sublayers once input shape is known
        self.flatten = td(layers.Flatten('channels_last'))
        self.projection = td(layers.Dense(self.token_size, use_bias=False))
        super().build(input_shape)  # mark as built

    def call(self, inputs, training=None, mask=None):
        x = self.flatten(inputs)
        return self.projection(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "token_size": self.token_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
