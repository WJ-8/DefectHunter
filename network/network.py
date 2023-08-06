from keras import backend as K
from keras import initializers
from keras.layers import Layer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.projection_dim = embedding_dim // num_heads
        self.query_dense = layers.Dense(embedding_dim)
        self.key_dense = layers.Dense(embedding_dim)
        self.value_dense = layers.Dense(embedding_dim)
        self.combine_heads = layers.Dense(embedding_dim)
        self.dropout_layer = layers.Dropout(dropout)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / (tf.math.sqrt(dim_key) + 1)

        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, _ = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(attention, (batch_size, -1, self.embedding_dim))
        output = self.combine_heads(concat_attention)
        output = self.dropout_layer(output)
        return output


class FeedForwardNetwork(layers.Layer):
    def __init__(self, embedding_dim, intermediate_dim):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = layers.Dense(intermediate_dim, activation="relu")
        self.dense2 = layers.Dense(embedding_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x


class ConformerEncoderBlock(layers.Layer):
    def __init__(self, embedding_dim, num_heads, feed_forward_dim, dropout=0.1):
        super(ConformerEncoderBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
        self.feed_forward_network = FeedForwardNetwork(embedding_dim, feed_forward_dim)
        self.attention_layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.feed_forward_layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training):
        attention_output = self.attention(inputs)
        attention_output = self.dropout(attention_output, training=training)
        residual = tf.add(inputs, attention_output)
        attention_output = self.attention_layer_norm(residual)

        feed_forward_output = self.feed_forward_network(attention_output)
        feed_forward_output = self.dropout(feed_forward_output, training=training)
        output = tf.add(attention_output, feed_forward_output)
        output = self.feed_forward_layer_norm(output)
        return output


class ConformerEncoder(layers.Layer):
    def __init__(self, num_blocks, embedding_dim, num_heads, feed_forward_dim, dropout=0.1):
        super(ConformerEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.encoders = [ConformerEncoderBlock(embedding_dim,
                                               num_heads,
                                               feed_forward_dim,
                                               dropout)
                         for _ in range(num_blocks)]
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training):
        x = inputs
        for i in range(self.num_blocks):
            x = self.encoders[i](x, training)
        x = self.dropout(x, training=training)
        return x

class MyMultiHeadAttention(Layer):
    def __init__(self, output_dim, num_head, kernel_initializer='glorot_uniform', **kwargs):
        self.output_dim = output_dim
        self.num_head = num_head
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(MyMultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(self.num_head, 3, input_shape[2], self.output_dim),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        self.Wo = self.add_weight(name='Wo',
                                  shape=(self.num_head * self.output_dim, self.output_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
        self.built = True

    def call(self, x):
        q = K.dot(x, self.W[0, 0])
        k = K.dot(x, self.W[0, 1])
        v = K.dot(x, self.W[0, 2])
        e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))  # 把k转置，并与q点乘 [0, 2, 1]宽度和高度交换
        e = e / (self.output_dim ** 0.5)
        e = K.softmax(e)
        outputs = K.batch_dot(e, v)
        for i in range(1, self.W.shape[0]):
            q = K.dot(x, self.W[i, 0])
            k = K.dot(x, self.W[i, 1])
            v = K.dot(x, self.W[i, 2])
            # print('q_shape:'+str(q.shape))
            e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))  # 把k转置，并与q点乘
            e = e / (self.output_dim ** 0.5)
            e = K.softmax(e)
            # print('e_shape:'+str(e.shape))
            o = K.batch_dot(e, v)
            outputs = K.concatenate([outputs, o])
        z = K.dot(outputs, self.Wo)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "output_dim": self.output_dim,
            'num_head': self.num_head,
            'kernel_initializer': self.kernel_initializer
        })
        return config


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions