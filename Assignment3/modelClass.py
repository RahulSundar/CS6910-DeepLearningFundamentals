import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers
 

#from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten, Activation, LSTM, SimpleRNN, GRU, TimeDistributed
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model, Sequential,  Model
from tensorflow.keras.callbacks import EarlyStopping


import wandb


class Translation():

    def __init__(self, modelConfigDict, srcVocab2Int, tgtVocab2Int, using_pretrained_model = False):
        #self.native_vocabulary = modelConfigDict["native_vocabulary"]
        self.numEncoders = modelConfigDict["numEncoders"]
        self.cell_type = modelConfigDict["cell_type"]
        self.latentDim = modelConfigDict["latentDim"]
        self.dropout = modelConfigDict["dropout"]
        self.numDecoders = modelConfigDict["numDecoders"]
        self.hidden = modelConfigDict["hidden"]
        self.tgtVocab2Int = tgtVocab2Int
        self.srcVocab2Int = srcVocab2Int

    def build_configurable_model(self):       
        if self.cell_type == "RNN":
            # encoder
            encoder_inputs = Input(shape=(None, len(srcVocab2Int)))
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
            decoder_inputs = Input(shape=(None, len(tgtVocab2Int)))
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
            decoder_dense = Dense(len(tgtVocab2Int), activation="softmax")
            decoder_outputs = decoder_dense(hidden_outputs)
            model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
            
            return model
        
        elif self.cell_type == "LSTM":
            # encoder
            encoder_inputs = Input(shape=(None, len(srcVocab2Int)))
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
            decoder_inputs = Input(shape=(None, len(tgtVocab2Int)))
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
            decoder_dense = Dense(len(tgtVocab2Int), activation="softmax")
            decoder_outputs = decoder_dense(hidden_outputs)
            model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
            
            return model
        
        elif self.cell_type == "GRU":
            # encoder
            encoder_inputs = Input(shape=(None, len(srcVocab2Int)))
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
            decoder_inputs = Input(shape=(None, len(tgtVocab2Int)))
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
            decoder_dense = Dense(len(tgtVocab2Int), activation="softmax")
            decoder_outputs = decoder_dense(hidden_outputs)
            model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
            
            return model
