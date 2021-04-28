import numpy as np
import tensorflow as tf
from tensorflow import keras

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
#Invalid device or cannot modify virtual devices once initialized.
    pass
 
 
batch_size = 32  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 60000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'te.translit.sampled.train.tsv'


input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
for line in lines[:min(60000, len(lines) - 1)]:
    target_text, input_text, _ = line.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    #encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    #decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    #decoder_target_data[i, t:, target_token_index[" "]] = 1.0
    
val_data_path = 'te.translit.sampled.dev.tsv'

val_input_texts = []
val_target_texts = []
val_input_characters = set()
val_target_characters = set()
with open(val_data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
for line in lines[:min(60000, len(lines) - 1)]:
    val_target_text, val_input_text, _ = line.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    val_target_text = "\t" + val_target_text + "\n"
    val_input_texts.append(input_text)
    val_target_texts.append(target_text)
    for char in val_input_text:
        if char not in val_input_characters:
            val_input_characters.add(char)
    for char in val_target_text:
        if char not in val_target_characters:
            val_target_characters.add(char)

val_input_characters = sorted(list(val_input_characters))
val_target_characters = sorted(list(val_target_characters))
num_val_encoder_tokens = len(val_input_characters)
num_val_decoder_tokens = len(val_target_characters)
max_val_encoder_seq_length = max([len(txt) for txt in val_input_texts])
max_val_decoder_seq_length = max([len(txt) for txt in val_target_texts])

print("Number of val samples:", len(val_input_texts))
print("Number of unique val input tokens:", num_val_encoder_tokens)
print("Number of unique val output tokens:", num_val_decoder_tokens)
print("Max sequence length for val inputs:", max_val_encoder_seq_length)
print("Max sequence length for val outputs:", max_val_decoder_seq_length)

val_input_token_index = dict([(char, i) for i, char in enumerate(val_input_characters)])
val_target_token_index = dict([(char, i) for i, char in enumerate(val_target_characters)])

val_encoder_input_data = np.zeros(
    (len(val_input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
val_decoder_input_data = np.zeros(
    (len(val_input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
val_decoder_target_data = np.zeros(
    (len(val_input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (val_input_text, val_target_text) in enumerate(zip(val_input_texts, val_target_texts)):
    for t, char in enumerate(val_input_text):
        val_encoder_input_data[i, t, input_token_index[char]] = 1.0
    #encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(val_target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        val_decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            val_decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    #decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    #decoder_target_data[i, t:, target_token_index[" "]] = 1.0

    
# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data),
)
# Save model
model.save("s2s")

