
import numpy as np
import pandas as pd
import os

from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten, Activation, LSTM, SimpleRNN, GRU, TimeDistributed, Concatenate

from dataProcessor import DataProcessing
from modelClass import S2STranslation
from attention import AttentionLayer


import wandb
from wandb.keras import WandbCallback


import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
#Invalid device or cannot modify virtual devices once initialized.
    pass
 

DATAPATH = "/home/ratnamaru/Documents/Acads/Courses/Sem6/CS6910-FDL/GITHUB/CS6910-DeepLearningFundamentals/Assignment3/dakshina_dataset_v1.0"

#By default source language is english and target lang is telugu
dataBase = DataProcessing(DATAPATH) 

MODELPATH = "./TrainedModels/BestModelWithoutAttention/BestModelWithoutAttention"
MODELPATHATT = "./TrainedModels/BestModelWithAttention/BestModelWithAttention"

model = load_model(MODELPATH)
modelatt = load_model(MODELPATHATT)
modelatt2 = load_model("./TrainedModels/GRUen1_te_1_rmsprop_1_0.2_64_256/")
config_best = {
        "cell_type": "LSTM",
        "latentDim": 256,
        "hidden": 64,
        "optimiser": "adam",
        "numEncoders": 2,
        "numDecoders": 1,
        "dropout": 0.1,
        "epochs": 20,
        "batch_size": 32,
    }
    
    
config_best_attention = {
        "cell_type": "RNN",
        "latentDim": 256,
        "hidden": 16,
        "optimiser": "rmsprop",
        "numEncoders": 1,
        "numDecoders": 1,
        "dropout": 0.1,
        "epochs": 10,
        "batch_size": 32,
    }
    
config_best_attention2 = {
        "cell_type": "GRU",
        "latentDim": 256,
        "hidden": 128,
        "optimiser": "rmsprop",
        "numEncoders": 1,
        "numDecoders": 1,
        "dropout": 0.2,
        "epochs": 10,
        "batch_size": 32,
    }
    
#In case one needs to rerun the model   
def train():

    config_defaults = {
        "cell_type": "LSTM",
        "latentDim": 256,
        "hidden": 128,
        "optimiser": "rmsprop",
        "numEncoders": 2,
        "numDecoders": 1,
        "dropout": 0.1,
        "epochs": 10,
        "batch_size": 32,
    }


    wandb.init(config=config_defaults,  project="CS6910-Assignment-3", entity="rahulsundar")
    config = wandb.config
    wandb.run.name = (
        str(config.cell_type)
        + dataBase.source_lang
        + str(config.numEncoders)
        + "_"
        + dataBase.target_lang
        + "_"
        + str(config.numDecoders)
        + "_"
        + config.optimiser
        + "_"
        + str(config.epochs)
        + "_"
        + str(config.dropout) 
        + "_"
        + str(config.batch_size)
        + "_"
        + str(config.latentDim)
    )
    wandb.run.save()

    modelInit = S2STranslation(config,srcChar2Int=dataBase.source_char2int, tgtChar2Int=dataBase.target_char2int)
    
    #model = modelInit.build_configurable_model()
    model = modelInit.build_attention_model()
    model.summary()

    model.compile(
        optimizer=config.optimiser,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    earlystopping = EarlyStopping(
        monitor="val_accuracy", min_delta=0.01, patience=5, verbose=2, mode="auto"
    )

    model.fit(
        [dataBase.train_encoder_input, dataBase.train_decoder_input],
        dataBase.train_decoder_target,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=([dataBase.val_encoder_input, dataBase.val_decoder_input], dataBase.val_decoder_target),
        callbacks=[earlystopping, WandbCallback()],
    )

    model.save(os.path.join("./TrainedModels", wandb.run.name))    

	acc = test_model(model, config.cell_type, config.numEncoders+1, dataBase.test_encoder_input, dataBase.test, dataBase.target_int2char, dataBase.target_char2int)
	print(f'Test Accuracy: {acc}')
	wandb.log({'test_accuracy': acc})

    wandb.finish()
    
    #return model



def test_model(
    model,
    attention = False
):

    if attention == False:
        wandb.init(config=config_best,  project="CS6910-Assignment-3", entity="rahulsundar")
        config = wandb.config
        wandb.run.name = (
            "Inference_" 
            + str(config.cell_type)
            + dataBase.source_lang
            + str(config.numEncoders)
            + "_"
            + dataBase.target_lang
            + "_"
            + str(config.numDecoders)
            + "_"
            + config.optimiser
            + "_"
            + str(config.epochs)
            + "_"
            + str(config.dropout) 
            + "_"
            + str(config.batch_size)
            + "_"
            + str(config.latentDim)
        )
        wandb.run.save()


        if config.cell_type == "LSTM":
            encoder_inputs = model.input[0]
            
            if config.numEncoders == 1:
                encoder_outputs, state_h_enc, state_c_enc = model.get_layer(name = "lstm").output 
            else:           
                encoder_outputs, state_h_enc, state_c_enc = model.get_layer(name = "lstm_"+ str(config.numEncoders-1)).output

            encoder_states = [state_h_enc, state_c_enc]
            encoder_model = Model(encoder_inputs, encoder_states)

            decoder_inputs = model.input[1]
            decoder_state_input_h = Input(shape=(config.latentDim,), name="input_3")
            decoder_state_input_c = Input(shape=(config.latentDim,), name="input_4")
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_lstm = model.layers[-3]
            decoder_outputs, state_h_dec, state_c_dec = decoder_lstm( decoder_inputs, initial_state=decoder_states_inputs )
            decoder_states = [state_h_dec, state_c_dec]
            decoder_dense = model.layers[-2]
            decoder_outputs = decoder_dense(decoder_outputs)
            
            decoder_dense = model.layers[-1]
            decoder_outputs = decoder_dense(decoder_outputs)
            decoder_model = Model(
                [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
            )
        elif config.cell_type == "GRU" or config.cell_type == "RNN":
            encoder_inputs = model.input[0]
            if config.cell_type == "GRU":
                if config.numEncoders == 1:
                    encoder_outputs, state = model.get_layer(name = "gru").output
                else:
                    encoder_outputs, state = model.get_layer(name = "gru_"+ str(config.numEncoders-1)).output
            else:
                if config.numEncoders == 1:
                    encoder_outputs, state = model.get_layer(name = "simple_rnn").output
                else:
                    encoder_outputs, state = model.get_layer(name = "simple_rnn_"+ str(config.numEncoders-1)).output

            encoder_states = [state]

            encoder_model = Model(encoder_inputs, encoder_states)

            decoder_inputs = model.input[1]

            decoder_state = Input(shape=(config.latentDim,), name="input_3")
            decoder_states_inputs = [decoder_state]

            decoder_gru = model.layers[-3]
            (decoder_outputs, state,) = decoder_gru(decoder_inputs, initial_state=decoder_states_inputs)
            decoder_states = [state]
            decoder_dense = model.layers[-2]
            decoder_outputs = decoder_dense(decoder_outputs)
            decoder_dense = model.layers[-1]
            decoder_outputs = decoder_dense(decoder_outputs)
            decoder_model = Model(
                [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
            )

        def decode_sequence(input_seq):
            # Encode the input as state vectors.
            states_value = encoder_model.predict(input_seq)

            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1, 1, len(dataBase.target_char2int)))
            # Populate the first character of target sequence with the start character.
            target_seq[0, 0, dataBase.target_char2int["\n"]] = 1.0

            # Sampling loop for a batch of sequences
            # (to simplify, here we assume a batch of size 1).
            stop_condition = False
            decoded_sentence = ""
            while not stop_condition:
                if config.cell_type == "LSTM":
                    output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
                elif config.cell_type == "RNN" or config.cell_type == "GRU":
                    states_value = states_value[0].reshape((1, 256))
                    output_tokens, h = decoder_model.predict([target_seq] + [states_value])

                # Sample a token
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_char = dataBase.target_int2char[sampled_token_index]
                decoded_sentence += sampled_char

                # Exit condition: either hit max length
                # or find stop character.
                if sampled_char == "\n" or len(decoded_sentence) > 25:
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1, len(dataBase.target_char2int)))
                target_seq[0, 0, sampled_token_index] = 1.0

                # Update states
                if config.cell_type == "LSTM":
                    states_value = [h, c]
                elif config.cell_type == "RNN" or config.cell_type == "GRU":
                    states_value = [h]
            return decoded_sentence

        acc = 0
        for i, row in dataBase.test.iterrows():
            input_seq = dataBase.test_encoder_input[i : i + 1]
            decoded_sentence = decode_sequence(input_seq)
            og_tokens = [dataBase.target_char2int[x] for x in row["tgt"]]
            predicted_tokens = [dataBase.target_char2int[x] for x in decoded_sentence.rstrip("\n")]
            # if decoded_sentence == row['tgt']:
            #   acc += 1

            if og_tokens == predicted_tokens:
                acc += 1

            if i % 100 == 0:
                print(f"Finished {i} examples")
                print(f"Source: {row['src']}")
                print(f"Original: {row['tgt']}")
                print(f"Predicted: {decoded_sentence}")
                print(f"Accuracy: {acc / (i+1)}")
                print(og_tokens)
                print(predicted_tokens)
                

        print(f'Test Accuracy: {acc}')
        wandb.log({'test_accuracy': acc / len(dataBase.test)})
        wandb.finish()
        return acc / len(dataBase.test)

    elif attention == True:
        wandb.init(config=config_best_attention2,  project="CS6910-Assignment-3", entity="rahulsundar")
        config = wandb.config
        wandb.run.name = (
            "Inference_WithAttn_" 
            + str(config.cell_type)
            + dataBase.source_lang
            + str(config.numEncoders)
            + "_"
            + dataBase.target_lang
            + "_"
            + str(config.numDecoders)
            + "_"
            + config.optimiser
            + "_"
            + str(config.epochs)
            + "_"
            + str(config.dropout) 
            + "_"
            + str(config.batch_size)
            + "_"
            + str(config.latentDim)
        )
        wandb.run.save()


        if config.cell_type == "LSTM":
            encoder_inputs = model.input[0]
            if config.numEncoders == 1:
                encoder_outputs, state_h_enc, state_c_enc = model.get_layer(name = "lstm").output 
            else:           
                encoder_outputs, state_h_enc, state_c_enc = model.get_layer(name = "lstm_"+ str(config.numEncoders-1)).output
            encoder_first_outputs, _, _ = model.get_layer(name = "lstm").output
            encoder_states = [state_h_enc, state_c_enc]
            encoder_model = Model(encoder_inputs, encoder_states)

            decoder_inputs = model.input[1]
            decoder_state_input_h = Input(shape=(config.latentDim,), name="input_3")
            decoder_state_input_c = Input(shape=(config.latentDim,), name="input_4")
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            #decoder_lstm = model.layers[-3]
            decoder_lstm = model.get_layer(name = "lstm_"+ str(config.numEncoders + config.numDecoders -1))
            decoder_outputs, state_h_dec, state_c_dec = decoder_lstm( decoder_inputs, initial_state=decoder_states_inputs )
            decoder_states = [state_h_dec, state_c_dec]

            attention_layer = model.get_layer(name = "attention_layer")#AttentionLayer(name='attention_layer')
            attention_out, attention_states = attention_layer([encoder_first_outputs, decoder_outputs])


            decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_out])
            
            decoder_dense = model.layers[-2]
            decoder_time = TimeDistributed(decoder_dense)
            hidden_outputs = decoder_time(decoder_concat_input)
            decoder_dense = model.layers[-1]
            decoder_outputs = decoder_dense(hidden_outputs)

            decoder_model = Model(inputs = [encoder_inputs, decoder_inputs] + decoder_states_inputs, outputs = [attention_states, decoder_outputs] +  decoder_states)
            #decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states )
            
        elif config.cell_type == "GRU" or config.cell_type == "RNN":
            encoder_inputs = model.input[0]
            if config.cell_type == "GRU":
                if config.numEncoders == 1:
                    encoder_outputs, state = model.get_layer(name = "gru").output
                else:
                    encoder_outputs, state = model.get_layer(name = "gru_"+ str(config.numEncoders-1)).output
                encoder_first_outputs, _ = model.get_layer(name = "gru").output
            else:
                if config.numEncoders == 1:
                    encoder_outputs, state = model.get_layer(name = "simple_rnn").output
                else:
                    encoder_outputs, state = model.get_layer(name = "simple_rnn_"+ str(config.numEncoders-1)).output
                encoder_first_outputs, _ = model.get_layer(name = "simple_rnn").output
            encoder_states = [state]

            encoder_model = Model(encoder_inputs, encoder_states)

            decoder_inputs = model.input[1]

            decoder_state = Input(shape=(config.latentDim,), name="input_3")
            decoder_states_inputs = [decoder_state]

            if config.cell_type == "GRU":
                decoder_gru = model.get_layer(name = "gru_"+ str(config.numEncoders + config.numDecoders -1))#model.layers[-3]
                (decoder_outputs, state) = decoder_gru(decoder_inputs, initial_state=decoder_states_inputs)
                decoder_states = [state]

            else:
                decoder_gru = model.get_layer(name = "simple_rnn_"+ str(config.numEncoders + config.numDecoders -1))#model.layers[-3]
                (decoder_outputs, state) = decoder_gru(decoder_inputs, initial_state=decoder_states_inputs)
                decoder_states = [state]

                    
            attention_layer = AttentionLayer(name='attention_layer')
            #decoder_outputs_att = decoder_ouputs
            attention_out, attention_states = attention_layer([encoder_first_outputs, decoder_outputs])

            decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_out])

            decoder_dense = model.layers[-2]
            decoder_time = TimeDistributed(decoder_dense)
            hidden_outputs = decoder_time(decoder_concat_input)
            decoder_dense = model.layers[-1]
            decoder_outputs = decoder_dense(hidden_outputs)

            decoder_model = Model(inputs = [encoder_inputs, decoder_inputs] + decoder_states_inputs, outputs = [attention_states, decoder_outputs] +  decoder_states)
        def decode_sequence(input_seq):
            # Encode the input as state vectors.
            encoder_first_outputs, _, states_value = encoder_model.predict(input_seq)

            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1, 1, len(dataBase.target_char2int)))
            # Populate the first character of target sequence with the start character.
            target_seq[0, 0, dataBase.target_char2int["\n"]] = 1.0

            # Sampling loop for a batch of sequences
            # (to simplify, here we assume a batch of size 1).
            stop_condition = False
            decoded_sentence = ""
            attention_weights = []
            while not stop_condition:
                if config.cell_type == "LSTM":
                    output_tokens, h, c = decoder_model.predict([target_seq, encoder_first_outputs] + states_value)
                elif config.cell_type == "RNN" or config.cell_type == "GRU":
                    states_value = states_value[0].reshape((1, config.latentDim))
                    output_tokens, h = decoder_model.predict([target_seq] + [encoder_first_outputs] + [states_value])
                dec_ind = np.argmax(output_tokens, axis=-1)[0, 0]
                attention_weights.append((dec_ind, attn_states))
                # Sample a token
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_char = dataBase.target_int2char[sampled_token_index]
                decoded_sentence += sampled_char

                # Exit condition: either hit max length
                # or find stop character.
                if sampled_char == "\n" or len(decoded_sentence) > 25:
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1, len(dataBase.target_char2int)))
                target_seq[0, 0, sampled_token_index] = 1.0

                # Update states
                if config.cell_type == "LSTM":
                    states_value = [h, c]
                elif config.cell_type == "RNN" or config.cell_type == "GRU":
                    states_value = [h]
            return decoded_sentence, attention_weights

        acc = 0
        sourcelang = []
        predictions = []
        original = []
        attention_weights_test = []
        for i, row in dataBase.test.iterrows():
            input_seq = dataBase.test_encoder_input[i : i + 1]
            decoded_sentence, attention_weights = decode_sequence(input_seq)
            og_tokens = [dataBase.target_char2int[x] for x in row["tgt"]]
            predicted_tokens = [dataBase.target_char2int[x] for x in decoded_sentence.rstrip("\n")]
            # if decoded_sentence == row['tgt']:
            #   acc += 1
            sourcelang.append(row['src'])
            original.append(row['tgt'])
            predictions.append(decoded_sentence)
            attention_weights_test.append(attention_weights)
            if og_tokens == predicted_tokens:
                acc += 1

            if i % 100 == 0:
                print(f"Finished {i} examples")
                print(f"Source: {row['src']}")
                print(f"Original: {row['tgt']}")
                print(f"Predicted: {decoded_sentence}")
                print(f"Accuracy: {acc / (i+1)}")
                print(og_tokens)
                print(predicted_tokens)
                

        print(f'Test Accuracy: {acc}')
        wandb.log({'test_accuracy': acc / len(dataBase.test)})
        wandb.finish()
        return acc / len(dataBase.test) , sourcelang, original, predictions, attention_weights_test


acc = test_model(model, config.cell_type, config.numEncoders+1, dataBase.test_encoder_input, dataBase.test, dataBase.target_int2char, dataBase.target_char2int)

