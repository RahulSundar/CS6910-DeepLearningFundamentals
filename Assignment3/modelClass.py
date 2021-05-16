import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers
 

#from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten, Activation, LSTM, SimpleRNN, GRU, TimeDistributed, Concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model, Sequential,  Model
from tensorflow.keras.callbacks import EarlyStopping
from attention import AttentionLayer

import wandb


class S2STranslation():

    def __init__(self, modelConfigDict, srcChar2Int, tgtChar2Int, using_pretrained_model = False):
        #self.native_vocabulary = modelConfigDict["native_vocabulary"]
        self.numEncoders = modelConfigDict["numEncoders"]
        self.cell_type = modelConfigDict["cell_type"]
        self.latentDim = modelConfigDict["latentDim"]
        self.dropout = modelConfigDict["dropout"]
        self.numDecoders = modelConfigDict["numDecoders"]
        self.hidden = modelConfigDict["hidden"]
        self.tgtChar2Int = tgtChar2Int
        self.srcChar2Int = srcChar2Int

    def build_configurable_model(self):       
        if self.cell_type == "RNN":
            # encoder
            encoder_inputs = Input(shape=(None, len(self.srcChar2Int)))
            encoder_outputs = encoder_inputs
            for i in range(1, self.numEncoders + 1):
                encoder = SimpleRNN(
                    self.latentDim,
                    return_state=True,
                    return_sequences=True,
                    dropout=self.dropout,
                )
                encoder_outputs, state = encoder(encoder_inputs)
            encoder_states = [state]

            # decoder
            decoder_inputs = Input(shape=(None, len(self.tgtChar2Int)))
            decoder_outputs = decoder_inputs
            for i in range(1, self.numDecoders + 1):
                decoder = SimpleRNN(
                    self.latentDim,
                    return_sequences=True,
                    return_state=True,
                    dropout=self.dropout,
                )
                decoder_outputs, _ = decoder(decoder_inputs, initial_state=encoder_states)

            # dense
            hidden = Dense(self.hidden, activation="relu")
            hidden_outputs = hidden(decoder_outputs)
            decoder_dense = Dense(len(self.tgtChar2Int), activation="softmax")
            decoder_outputs = decoder_dense(hidden_outputs)
            model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
            
            return model
        
        elif self.cell_type == "LSTM":
            # encoder
            encoder_inputs = Input(shape=(None, len(self.srcChar2Int)))
            encoder_outputs = encoder_inputs
            for i in range(1, self.numEncoders + 1):
                encoder = LSTM(
                    self.latentDim,
                    return_state=True,
                    return_sequences=True,
                    dropout=self.dropout,
                )
                encoder_outputs, state_h, state_c = encoder(encoder_outputs)
            encoder_states = [state_h, state_c]

            # decoder
            decoder_inputs = Input(shape=(None, len(self.tgtChar2Int)))
            decoder_outputs = decoder_inputs
            for i in range(1, self.numDecoders + 1):
                decoder = LSTM(
                    self.latentDim,
                    return_state=True,
                    return_sequences=True,
                    dropout=self.dropout,
                )
                decoder_outputs, _, _ = decoder(
                    decoder_outputs, initial_state=encoder_states
                )

            # dense
            hidden = Dense(self.hidden, activation="relu")
            hidden_outputs = hidden(decoder_outputs)
            decoder_dense = Dense(len(self.tgtChar2Int), activation="softmax")
            decoder_outputs = decoder_dense(hidden_outputs)
            model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
            
            return model
        
        elif self.cell_type == "GRU":
            # encoder
            encoder_inputs = Input(shape=(None, len(self.srcChar2Int)))
            encoder_outputs = encoder_inputs
            for i in range(1, self.numEncoders + 1):
                encoder = GRU(
                    self.latentDim,
                    return_state=True,
                    return_sequences=True,
                    dropout=self.dropout,
                )
                encoder_outputs, state = encoder(encoder_inputs)
            encoder_states = [state]

            # decoder
            decoder_inputs = Input(shape=(None, len(self.tgtChar2Int)))
            decoder_outputs = decoder_inputs
            for i in range(1, self.numDecoders + 1):
                decoder = GRU(
                    self.latentDim,
                    return_sequences=True,
                    return_state=True,
                    dropout=self.dropout,
                )
                decoder_outputs, _ = decoder(decoder_inputs, initial_state=encoder_states)

            # dense
            hidden = Dense(self.hidden, activation="relu")
            hidden_outputs = hidden(decoder_outputs)
            decoder_dense = Dense(len(self.tgtChar2Int), activation="softmax")
            decoder_outputs = decoder_dense(hidden_outputs)
            model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
            
            return model
            
            
    def build_attention_model(self):       
        if self.cell_type == "RNN":
            # encoder
            encoder_inputs = Input(shape=(None, len(self.srcChar2Int)))
            encoder_outputs = encoder_inputs
            for i in range(1, self.numEncoders + 1):
                encoder = SimpleRNN(
                    self.latentDim,
                    return_state=True,
                    return_sequences=True,
                    dropout=self.dropout,
                )
                encoder_outputs, state = encoder(encoder_inputs) 
                
                if i == 1:
                    encoder_first_outputs= encoder_outputs                  
            encoder_states = [state]
            

            # decoder
            decoder_inputs = Input(shape=(None, len(self.tgtChar2Int)))
            decoder_outputs = decoder_inputs
            for i in range(1, self.numDecoders + 1):
                decoder = SimpleRNN(
                    self.latentDim,
                    return_sequences=True,
                    return_state=True,
                    dropout=self.dropout,
                )
                decoder_outputs, _ = decoder(decoder_inputs, initial_state=encoder_states)
                
                if i == self.numDecoders:
                    decoder_first_outputs = decoder_outputs

            attention_layer = AttentionLayer(name='attention_layer')
            attention_out, attention_states = attention_layer([encoder_first_outputs, decoder_first_outputs])


            decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_out])

            # dense
            hidden = Dense(self.hidden, activation="relu")
            hidden_time = TimeDistributed(hidden, name='time_distributed_layer')
            hidden_outputs = hidden(decoder_concat_input)
            decoder_dense = Dense(len(self.tgtChar2Int), activation="softmax")
            decoder_outputs = decoder_dense(hidden_outputs)
            model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
            
            return model
        
        elif self.cell_type == "LSTM":
            # encoder
            encoder_inputs = Input(shape=(None, len(self.srcChar2Int)))
            encoder_outputs = encoder_inputs
            for i in range(1, self.numEncoders + 1):
                encoder = LSTM(
                    self.latentDim,
                    return_state=True,
                    return_sequences=True,
                    dropout=self.dropout,
                )
                encoder_outputs, state_h, state_c = encoder(encoder_outputs)
                if i == 1:
                    encoder_first_outputs= encoder_outputs                  
         
            encoder_states = [state_h, state_c]

            # decoder
            decoder_inputs = Input(shape=(None, len(self.tgtChar2Int)))
            decoder_outputs = decoder_inputs
            for i in range(1, self.numDecoders + 1):
                decoder = LSTM(
                    self.latentDim,
                    return_state=True,
                    return_sequences=True,
                    dropout=self.dropout,
                )
                decoder_outputs, _, _ = decoder(
                    decoder_outputs, initial_state=encoder_states
                )
                if i == self.numDecoders:
                    decoder_first_outputs = decoder_outputs

            attention_layer = AttentionLayer(name='attention_layer')
            attention_out, attention_states = attention_layer([encoder_first_outputs, decoder_first_outputs])

            decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_out])

            # dense
            hidden = Dense(self.hidden, activation="relu")
            hidden_time = TimeDistributed(hidden, name='time_distributed_layer')
            hidden_outputs = hidden(decoder_concat_input)
            decoder_dense = Dense(len(self.tgtChar2Int), activation="softmax")
            decoder_outputs = decoder_dense(hidden_outputs)
            model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
            
            return model
        
        elif self.cell_type == "GRU":
            # encoder
            encoder_inputs = Input(shape=(None, len(self.srcChar2Int)))
            encoder_outputs = encoder_inputs
            for i in range(1, self.numEncoders + 1):
                encoder = GRU(
                    self.latentDim,
                    return_state=True,
                    return_sequences=True,
                    dropout=self.dropout,
                )
                encoder_outputs, state = encoder(encoder_inputs)

                if i == 1:
                    encoder_first_outputs= encoder_outputs                  
         
            encoder_states = [state]

            # decoder
            decoder_inputs = Input(shape=(None, len(self.tgtChar2Int)))
            decoder_outputs = decoder_inputs
            for i in range(1, self.numDecoders + 1):
                decoder = GRU(
                    self.latentDim,
                    return_sequences=True,
                    return_state=True,
                    dropout=self.dropout,
                )
                decoder_outputs, _ = decoder(decoder_inputs, initial_state=encoder_states)
                if i == self.numDecoders:
                    decoder_first_outputs = decoder_outputs



            attention_layer = AttentionLayer(name='attention_layer')
            attention_out, attention_states = attention_layer([encoder_first_outputs, decoder_first_outputs])

            decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_out])

            # dense
            hidden = Dense(self.hidden, activation="relu")
            hidden_time = TimeDistributed(hidden, name='time_distributed_layer')
            hidden_outputs = hidden(decoder_concat_input)
            decoder_dense = Dense(len(self.tgtChar2Int), activation="softmax")
            decoder_outputs = decoder_dense(hidden_outputs)
            model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
            
            return model
