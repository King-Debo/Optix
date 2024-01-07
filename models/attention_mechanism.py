# models/attention_mechanism.py
# This file contains the code to implement the attention mechanism method for option chain price prediction, using the TensorFlow and the Keras libraries.

# Import the libraries
import tensorflow as tf
from tensorflow import keras

# Define the function to implement the attention mechanism method
def attention_mechanism(data):
    # Get the input and output variables from the data
    X = data[["underlying_price", "strike_price", "interest_rate", "dividend_rate", "expiration_date", "call_iv", "put_iv"]] # The input variables
    y = data[["call_ltp", "put_ltp"]] # The output variables

    # Reshape the input variables to a three-dimensional tensor
    X = X.values.reshape((X.shape[0], 1, X.shape[1]))

    # Define the attention layer
    class AttentionLayer(keras.layers.Layer):
        def __init__(self, units):
            super(AttentionLayer, self).__init__()
            self.W1 = keras.layers.Dense(units)
            self.W2 = keras.layers.Dense(units)
            self.V = keras.layers.Dense(1)

        def call(self, inputs):
            # inputs is a tuple of (encoder_output, decoder_hidden_state)
            encoder_output, decoder_hidden_state = inputs
            # Expand the decoder hidden state to match the encoder output shape
            decoder_hidden_state = tf.expand_dims(decoder_hidden_state, 1)
            # Calculate the attention score
            score = self.V(tf.nn.tanh(self.W1(encoder_output) + self.W2(decoder_hidden_state)))
            # Apply the softmax function to get the attention weights
            attention_weights = tf.nn.softmax(score, axis=1)
            # Multiply the encoder output with the attention weights
            context_vector = attention_weights * encoder_output
            # Sum the context vector along the time axis
            context_vector = tf.reduce_sum(context_vector, axis=1)
            # Return the context vector and the attention weights
            return context_vector, attention_weights

    # Define the encoder
    class Encoder(keras.Model):
        def __init__(self, units):
            super(Encoder, self).__init__()
            self.units = units
            self.lstm = keras.layers.LSTM(units, return_sequences=True, return_state=True)

        def call(self, inputs):
            # inputs is the encoder input
            # Pass the inputs to the LSTM layer
            encoder_output, encoder_hidden_state, encoder_cell_state = self.lstm(inputs)
            # Return the encoder output and the encoder hidden state and cell state
            return encoder_output, encoder_hidden_state, encoder_cell_state

    # Define the decoder
    class Decoder(keras.Model):
        def __init__(self, units):
            super(Decoder, self).__init__()
            self.units = units
            self.lstm = keras.layers.LSTM(units, return_sequences=True, return_state=True)
            self.dense = keras.layers.Dense(2, activation="linear")
            self.attention = AttentionLayer(units)

        def call(self, inputs):
            # inputs is a tuple of (encoder_output, decoder_input, decoder_hidden_state, decoder_cell_state)
            encoder_output, decoder_input, decoder_hidden_state, decoder_cell_state = inputs
            # Pass the decoder input to the LSTM layer
            decoder_output, decoder_hidden_state, decoder_cell_state = self.lstm(decoder_input, initial_state=[decoder_hidden_state, decoder_cell_state])
            # Apply the attention mechanism to get the context vector and the attention weights
            context_vector, attention_weights = self.attention((encoder_output, decoder_hidden_state))
            # Concatenate the context vector and the decoder output
            decoder_output = tf.concat([context_vector, decoder_output], axis=-1)
            # Pass the concatenated vector to the dense layer
            decoder_output = self.dense(decoder_output)
            # Return the decoder output and the decoder hidden state and cell state
            return decoder_output, decoder_hidden_state, decoder_cell_state

    # Define the encoder and decoder units
    units = 64

    # Define the encoder and decoder objects
    encoder = Encoder(units)
    decoder = Decoder(units)

    # Define the optimizer and the loss function
    optimizer = keras.optimizers.Adam()
    loss_object = keras.losses.MeanSquaredError()

    # Define the train step function
    @tf.function
    def train_step(inputs, targets):
        # Initialize the loss and the batch size
        loss = 0
        batch_size = inputs.shape[0]

        # Use the gradient tape to record the operations
        with tf.GradientTape() as tape:
            # Encode the inputs and get the encoder output and state
            encoder_output, encoder_hidden_state, encoder_cell_state = encoder(inputs)
            # Initialize the decoder input, hidden state, and cell state
            decoder_input = tf.expand_dims(tf.zeros(batch_size), 1)
            decoder_hidden_state = encoder_hidden_state
            decoder_cell_state = encoder_cell_state
            # Loop through the target sequence
            for t in range(targets.shape[1]):
                # Decode the decoder input and get the decoder output and state
                decoder_output, decoder_hidden_state, decoder_cell_state = decoder((encoder_output, decoder_input, decoder_hidden_state, decoder_cell_state))
                # Calculate the loss for the current time step
                loss += loss_object(targets[:, t], decoder_output[:, 0])
                # Update the decoder input with the target value
                decoder_input = tf.expand_dims(targets[:, t], 1)
        # Calculate the gradients
        gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
        # Apply the gradients to the optimizer
        optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
        # Return the loss
        return loss / targets.shape[1]

    # Define the number of epochs
    epochs = 100

    # Loop through the epochs
    for epoch in range(epochs):
        # Initialize the epoch loss
        epoch_loss = 0
        # Loop through the batches
        for i in range(0, X.shape[0], 32):
            # Get the input and output batch
            input_batch = X[i:i+32]
            output_batch = y[i:i+32]
            # Perform the train step and get the batch loss
            batch_loss = train_step(input_batch, output_batch)
            # Update the epoch loss
            epoch_loss += batch_loss
        # Print the epoch loss
        print(f"Epoch {epoch + 1}, Loss {epoch_loss.numpy():.4f}")

    # Define the predict function
    def predict(inputs):
        # Initialize the predictions
        predictions = np.zeros((inputs.shape[0], 2))
        # Encode the inputs and get the encoder output and state
        encoder_output, encoder_hidden_state, encoder_cell_state = encoder(inputs)
        # Initialize the decoder input, hidden state, and cell state
        decoder_input = tf.expand_dims(tf.zeros(inputs.shape[0]), 1)
        decoder_hidden_state = encoder_hidden_state
        decoder_cell_state = encoder_cell_state
        # Loop through the prediction sequence
        for t in range(2):
            # Decode the decoder input and get the decoder output and state
            decoder_output, decoder_hidden_state, decoder_cell_state = decoder((encoder_output, decoder_input, decoder_hidden_state, decoder_cell_state))
            # Store the decoder output in the predictions
            predictions[:, t] = decoder_output[:, 0]
            # Update the decoder input with the decoder output
            decoder_input = tf.expand_dims(decoder_output[:, 0], 1)
        # Return the predictions
        return predictions

    # Predict the call and put option prices
    y_pred = predict(X)

    # Return the predicted call and put option prices
    return y_pred[:, 0], y_pred[:, 1]
