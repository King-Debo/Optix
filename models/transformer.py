# models/transformer.py
# This file contains the code to implement the transformer method for option chain price prediction, using the TensorFlow and the Keras libraries.

# Import the libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Define the function to implement the transformer method
def transformer(data):
    # Get the input and output variables from the data
    X = data[["underlying_price", "strike_price", "interest_rate", "dividend_rate", "expiration_date", "call_iv", "put_iv"]] # The input variables
    y = data[["call_ltp", "put_ltp"]] # The output variables

    # Reshape the input variables to a three-dimensional tensor
    X = X.values.reshape((X.shape[0], 1, X.shape[1]))

    # Define the positional encoding layer
    class PositionalEncodingLayer(keras.layers.Layer):
        def __init__(self, d_model, max_len):
            super(PositionalEncodingLayer, self).__init__()
            self.d_model = d_model
            self.max_len = max_len
            self.pos_encoding = self.positional_encoding()

        def get_angles(self, pos, i):
            # Calculate the angles for the positional encoding
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / self.d_model)
            return pos * angle_rates

        def positional_encoding(self):
            # Calculate the positional encoding matrix
            angle_rads = self.get_angles(np.arange(self.max_len)[:, np.newaxis],
                                         np.arange(self.d_model)[np.newaxis, :])
            # Apply sine to even indices in the array
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
            # Apply cosine to odd indices in the array
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            # Expand the dimensions of the array
            pos_encoding = angle_rads[np.newaxis, ...]
            # Return the positional encoding matrix
            return tf.cast(pos_encoding, dtype=tf.float32)

        def call(self, inputs):
            # Add the positional encoding to the inputs
            return inputs + self.pos_encoding[:, :inputs.shape[1], :]

    # Define the multi-head attention layer
    class MultiHeadAttentionLayer(keras.layers.Layer):
        def __init__(self, d_model, num_heads):
            super(MultiHeadAttentionLayer, self).__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.depth = d_model // num_heads
            self.wq = keras.layers.Dense(d_model)
            self.wk = keras.layers.Dense(d_model)
            self.wv = keras.layers.Dense(d_model)
            self.dense = keras.layers.Dense(d_model)

        def split_heads(self, x, batch_size):
            # Split the last dimension into (num_heads, depth)
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            # Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
            return tf.transpose(x, perm=[0, 2, 1, 3])

        def scaled_dot_product_attention(self, q, k, v, mask):
            # Calculate the dot product of q and k
            matmul_qk = tf.matmul(q, k, transpose_b=True)
            # Scale the dot product by the square root of the depth
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
            # Add the mask to the scaled tensor
            if mask is not None:
                scaled_attention_logits += (mask * -1e9)
            # Apply the softmax function to get the attention weights
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            # Multiply the attention weights with v
            output = tf.matmul(attention_weights, v)
            # Return the output and the attention weights
            return output, attention_weights

        def call(self, inputs):
            # inputs is a tuple of (q, k, v, mask)
            q, k, v, mask = inputs
            # Get the batch size
            batch_size = tf.shape(q)[0]
            # Pass q, k, and v to the linear layers
            q = self.wq(q)
            k = self.wk(k)
            v = self.wv(v)
            # Split q, k, and v into multiple heads
            q = self.split_heads(q, batch_size)
            k = self.split_heads(k, batch_size)
            v = self.split_heads(v, batch_size)
            # Apply the scaled dot product attention
            scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
            # Concatenate the attention outputs
            scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
            concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
            # Pass the concatenated outputs to the linear layer
            output = self.dense(concat_attention)
            # Return the output and the attention weights
            return output, attention_weights

    # Define the point wise feed forward network
    def point_wise_feed_forward_network(d_model, dff):
        # Return a sequential model of two dense layers
        return keras.Sequential([
            keras.layers.Dense(dff, activation="relu"),
            keras.layers.Dense(d_model)
        ])

    # Define the encoder layer
    class EncoderLayer(keras.layers.Layer):
        def __init__(self, d_model, num_heads, dff, rate=0.1):
            super(EncoderLayer, self).__init__()
            self.mha = MultiHeadAttentionLayer(d_model, num_heads)
            self.ffn = point_wise_feed_forward_network(d_model, dff)
            self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = keras.layers.Dropout(rate)
            self.dropout2 = keras.layers.Dropout(rate)

        def call(self, inputs):
            # inputs is a tuple of (x, training, mask)
            x, training, mask = inputs
            # Apply the multi-head attention and the dropout
            attn_output, _ = self.mha((x, x, x, mask))
            attn_output = self.dropout1(attn_output, training=training)
            # Add and normalize the attention output and the input
            out1 = self.layernorm1(x + attn_output)
            # Apply the feed forward network and the dropout
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            # Add and normalize the feed forward output and the output 1
            out2 = self.layernorm2(out1 + ffn_output)
            # Return the output 2
            return out2

    # Define the decoder layer
    class DecoderLayer(keras.layers.Layer):
        def __init__(self, d_model, num_heads, dff, rate=0.1):
            super(DecoderLayer, self).__init__()
            self.mha1 = MultiHeadAttentionLayer(d_model, num_heads)
            self.mha2 = MultiHeadAttentionLayer(d_model, num_heads)
            self.ffn = point_wise_feed_forward_network(d_model, dff)
            self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = keras.layers.Dropout(rate)
            self.dropout2 = keras.layers.Dropout(rate)
            self.dropout3 = keras.layers.Dropout(rate)

        def call(self, inputs):
            # inputs is a tuple of (x, encoder_output, training, look_ahead_mask, padding_mask)
            x, encoder_output, training, look_ahead_mask, padding_mask = inputs
            # Apply the first multi-head attention and the dropout
            attn1, attn_weights_block1 = self.mha1((x, x, x, look_ahead_mask))
            attn1 = self.dropout1(attn1, training=training)
            # Add and normalize the attention output and the input
            out1 = self.layernorm1(attn1 + x)
            # Apply the second multi-head attention and the dropout
            attn2, attn_weights_block2 = self.mha2((out1, encoder_output, encoder_output, padding_mask))
            attn2 = self.dropout2(attn2, training=training)
            # Add and normalize the attention output and the output 1
            out2 = self.layernorm2(out1 + attn2)
            # Apply the feed forward network and the dropout
            ffn_output = self.ffn(out2)
            ffn_output = self.dropout3(ffn_output, training=training)
            # Add and normalize the feed forward output and the output 2
            out3 = self.layernorm3(out2 + ffn_output)
            # Return the output 3 and the attention weights
            return out3, attn_weights_block1, attn_weights_block2
