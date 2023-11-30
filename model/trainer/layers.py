import keras
import tensorflow as tf

from trainer.utils import positional_encoding


@keras.saving.register_keras_serializable(package="ChessformerLayers")
class InputEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim)

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embedding_dim": self.embedding_dim,
            }
        )
        return config

    def call(self, x):
        x = self.embedding_layer(x)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        return x


@keras.saving.register_keras_serializable(package="ChessformerLayers")
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        # Length has to be greater than or equal to actual sequence length
        self.positional_embedding = positional_encoding(
            length=2048, depth=embedding_dim)

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "embedding_dim": self.embedding_dim,
            }
        )
        return config
        
    def call(self, x):
        sequence_length = tf.shape(x)[1]
        x += self.positional_embedding[tf.newaxis, :sequence_length, :]
        return x


@keras.saving.register_keras_serializable(package="ChessformerLayers")
class DotProductAttention(tf.keras.layers.Layer):
    def __init__(self, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
 
    def call(self, queries, keys, values, d_k, training):
        d_k = tf.cast(d_k, tf.float32)
        scores = tf.linalg.matmul(queries, keys, transpose_b=True)
        scores /= tf.math.sqrt(d_k)
        weights = tf.nn.softmax(scores)
        weights = self.dropout_layer(weights, training)
        return tf.linalg.matmul(weights, values)


@keras.saving.register_keras_serializable(package="ChessformerLayers")
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_k, d_v, d_out, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_out = d_out
        self.dropout_rate = dropout_rate
        self.attention_layer = DotProductAttention(dropout_rate)
        # The dimension is d_k * num_heads so we can
        # perform all computations in parallel
        self.W_q = tf.keras.layers.Dense(d_k * num_heads)
        self.W_k = tf.keras.layers.Dense(d_k * num_heads)
        self.W_v = tf.keras.layers.Dense(d_v * num_heads)
        self.W_o = tf.keras.layers.Dense(d_out)

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "num_heads": self.num_heads,
                "d_k": self.d_k,
                "d_v": self.d_v,
                "d_out": self.d_out,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def reshape_qkv(self, x, reverse=False):
        if not reverse:
            # x.shape == (num_batches, sequence_length, d * num_heads)
            # where d in {d_k, d_v}
            num_batches, sequence_length = tf.shape(x)[0], tf.shape(x)[1]
            x = tf.reshape(
                    x, shape=(num_batches, sequence_length, self.num_heads, -1))
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            # Now x.shape == (num_batches, num_heads, sequence_length, d)
        else:
            num_batches, num_heads = tf.shape(x)[0], tf.shape(x)[1]
            sequence_length, d = tf.shape(x)[2], tf.shape(x)[3]
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            x = tf.reshape(
                    x, shape=(num_batches, sequence_length, d * num_heads))
        return x

    def call(self, queries, keys, values, training=None):
        q_reshaped = self.reshape_qkv(self.W_q(queries))
        k_reshaped = self.reshape_qkv(self.W_k(keys))
        v_reshaped = self.reshape_qkv(self.W_v(values))

        output_reshaped = self.attention_layer(
                q_reshaped, k_reshaped, v_reshaped, self.d_k, training)
        output = self.reshape_qkv(output_reshaped, reverse=True)

        return self.W_o(output)


@keras.saving.register_keras_serializable(package="ChessformerLayers")
class GlobalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_k, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_k = d_k
        self.dropout_rate = dropout_rate
        self.multihead_attention_layer = MultiHeadAttention(
                num_heads, d_k, d_k, d_k, dropout_rate)
        self.add_layer = tf.keras.layers.Add()
        self.normalization_layer = tf.keras.layers.LayerNormalization()

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "num_heads": self.num_heads,
                "d_k": self.d_k,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def call(self, x):
        attention_output = self.multihead_attention_layer(
            queries=x, keys=x, values=x)
        x = self.add_layer([x, attention_output])
        x = self.normalization_layer(x)
        return x


@keras.saving.register_keras_serializable(package="ChessformerLayers")
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_k, dff, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.d_k = d_k
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_k),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add_layer = tf.keras.layers.Add()
        self.normalization_layer = tf.keras.layers.LayerNormalization()

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "d_k": self.d_k,
                "dff": self.dff,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def call(self, x):
        x = self.add_layer([x, self.seq(x)])
        x = self.normalization_layer(x)
        return x


@keras.saving.register_keras_serializable(package="ChessformerLayers")
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_k, dff, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_k = d_k
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.self_attention_layer = GlobalSelfAttention(
                num_heads, d_k, dropout_rate)
        self.ffn = FeedForward(d_k, dff, dropout_rate)

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "num_heads": self.num_heads,
                "d_k": self.d_k,
                "dff": self.dff,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def call(self, x):
        x = self.self_attention_layer(x)
        x = self.ffn(x)
        return x


@keras.saving.register_keras_serializable(package="ChessformerLayers")
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, vocab_size, d_k, num_heads, dff,
                 dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.d_k = d_k
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.input_embedding_layer = InputEmbedding(vocab_size, d_k)
        self.positional_embedding_layer = PositionalEmbedding(d_k)
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.encoder_layers = tf.keras.Sequential([
            EncoderLayer(num_heads, d_k, dff, dropout_rate)
            for _ in range(num_layers)
        ])

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "num_layers": self.num_layers,
                "vocab_size": self.vocab_size,
                "d_k": self.d_k,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def call(self, x):
        x = self.input_embedding_layer(x)
        x = self.positional_embedding_layer(x)
        x = self.dropout_layer(x)
        x = self.encoder_layers(x)
        return x


@keras.saving.register_keras_serializable(package="ChessformerLayers")
class EloRegression(tf.keras.layers.Layer):
    def __init__(self, dff, **kwargs):
        super().__init__(**kwargs)
        self.dff = dff
        self.hidden_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.LayerNormalization(),
        ])
        # We will output two elo scores
        self.output_layer = tf.keras.layers.Dense(2)

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "dff": self.dff,
            }
        )
        return config

    def call(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


@keras.saving.register_keras_serializable(package="ChessformerLayers")
class ResultClassification(tf.keras.layers.Layer):
    def __init__(self, dff, **kwargs):
        super().__init__(**kwargs)
        self.dff = dff
        self.hidden_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.Dense(dff, activation="relu"),
        ])
        # There are three possible results
        self.output_layer = tf.keras.layers.Dense(3, activation="softmax")

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "dff": self.dff,
            }
        )
        return config

    def call(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
