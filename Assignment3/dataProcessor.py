import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import pathlib

from  matplotlib import pyplot as plt
import matplotlib.image as mpimg

class DataProcess():

    def __init__(self, DATAPATH, train_data, val_data, test_data, source_lang, target_lang):
    
        self.source_lang = source_lang
        self.target_lang = target_lang
    
        self.train = pd.read_csv(
            "dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.train.tsv",
            sep="\t",
            names=["tgt", "src", "count"],
        )
        self.dev = pd.read_csv(
            "dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.dev.tsv",
            sep="\t",
            names=["tgt", "src", "count"],
        )
        self.test = pd.read_csv(
            "dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.test.tsv",
            sep="\t",
            names=["tgt", "src", "count"],
        )

        # create train data
        train_data = preprocess(train["src"].to_list(), train["tgt"].to_list())
        (
            train_encoder_input,
            train_decoder_input,
            train_decoder_target,
            source_vocab,
            target_vocab,
        ) = train_data
        source_vocab2int, source_int2vocab = source_vocab
        target_vocab2int, target_int2vocab = target_vocab

        # create dev data
        dev_data = encode(
            dev["src"].to_list(),
            dev["tgt"].to_list(),
            list(source_vocab2int.keys()),
            list(target_vocab2int.keys()),
            source_vocab2int=source_vocab2int,
            target_vocab2int=target_vocab2int,
        )
        dev_encoder_input, dev_decoder_input, dev_decoder_target = dev_data
        source_vocab2int, source_int2vocab = source_vocab
        target_vocab2int, target_int2vocab = target_vocab

        # create test data
        test_data = encode(
            test["src"].to_list(),
            test["tgt"].to_list(),
            list(source_vocab2int.keys()),
            list(target_vocab2int.keys()),
            source_vocab2int=source_vocab2int,
            target_vocab2int=target_vocab2int,
        )
        test_encoder_input, test_decoder_input, test_decoder_target = test_data
        source_vocab2int, source_int2vocab = source_vocab
        target_vocab2int, target_int2vocab = target_vocab

    


    def dictionary_lookup(self, vocab):
        char2int = dict([(char, i) for i, char in enumerate(vocab)])
        int2char = dict((i, char) for char, i in char2int.items())
        return char2int, int2char


    def encode(self, source, target, source_chars, target_chars, source_char2int=None, target_char2int=None):
        num_encoder_tokens = len(source_chars)
        num_decoder_tokens = len(target_chars)
        max_source_length = max([len(txt) for txt in source])
        max_target_length = max([len(txt) for txt in target])

        source_vocab, target_vocab = None, None
        if source_char2int == None and target_char2int == None:
            print("Generating the dictionary lookups for character to integer mapping and back")
            source_char2int, source_int2char = dictionary_lookup(source_chars)
            target_char2int, target_int2char = dictionary_lookup(target_chars)

            source_vocab = (source_char2int, source_int2char)
            target_vocab = (target_char2int, target_int2char)

        encoder_input_data = np.zeros(
            (len(source), max_source_length, num_encoder_tokens), dtype="float32"
        )
        decoder_input_data = np.zeros(
            (len(source), max_target_length, num_decoder_tokens), dtype="float32"
        )
        decoder_target_data = np.zeros(
            (len(source), max_target_length, num_decoder_tokens), dtype="float32"
        )

        for i, (input_text, target_text) in enumerate(zip(source, target)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, source_char2int[char]] = 1.0
            encoder_input_data[i, t + 1 :, source_char2int[" "]] = 1.0
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, target_char2int[char]] = 1.0
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, target_char2int[char]] = 1.0
            decoder_input_data[i, t + 1 :, target_char2int[" "]] = 1.0
            decoder_target_data[i, t:, target_char2int[" "]] = 1.0
        if source_vocab != None and target_vocab != None:
            return (
                encoder_input_data,
                decoder_input_data,
                decoder_target_data,
                source_vocab,
                target_vocab,
            )
        else:
            return encoder_input_data, decoder_input_data, decoder_target_data


    def preprocess(self):
        source_chars = set()
        target_chars = set()

        source = [str(x) for x in self.source]
        target = [str(x) for x in self.target]

        source_words = []
        target_words = []
        for src, tgt in zip(source, target):
            tgt = "\t" + tgt + "\n"
            source_words.append(src)
            target_words.append(tgt)
            for char in src:
                if char not in source_chars:
                    source_chars.add(char)
            for char in tgt:
                if char not in target_chars:
                    target_chars.add(char)

        source_chars = sorted(list(source_chars))
        target_chars = sorted(list(target_chars))

        #The space needs to be appended so that the encode function doesn't throw errors
        source_chars.append(" ")
        target_chars.append(" ")

        num_encoder_tokens = len(source_chars)
        num_decoder_tokens = len(target_chars)
        max_source_length = max([len(txt) for txt in source_words])
        max_target_length = max([len(txt) for txt in target_words])

        print("Number of samples:", len(source))
        print("Source Vocab length:", num_encoder_tokens)
        print("Target Vocab length:", num_decoder_tokens)
        print("Max sequence length for inputs:", max_source_length)
        print("Max sequence length for outputs:", max_target_length)

        return encode(source_words, target_words, source_chars, target_chars)

