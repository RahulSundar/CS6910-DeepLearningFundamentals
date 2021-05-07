import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import pathlib


class DataProcessing():

    def __init__(self, DATAPATH, source_lang = 'en', target_lang = "te"):
    
        self.source_lang = source_lang
        self.target_lang = target_lang
    
        self.trainpath = os.path.join(DATAPATH, target_lang, "lexicons", target_lang+".translit.sampled.train.tsv")
        self.valpath = os.path.join(DATAPATH, target_lang, "lexicons", target_lang+".translit.sampled.dev.tsv")
        self.testpath = os.path.join(DATAPATH, target_lang, "lexicons", target_lang+".translit.sampled.test.tsv")
        self.train = pd.read_csv(
            self.trainpath,
            sep="\t",
            names=["tgt", "src", "count"],
        )
        self.val = pd.read_csv(
            self.valpath,
            sep="\t",
            names=["tgt", "src", "count"],
        )
        self.test = pd.read_csv(
            self.testpath,
            sep="\t",
            names=["tgt", "src", "count"],
        )

        # create train data
        self.train_data = self.preprocess(self.train["src"].to_list(), self.train["tgt"].to_list())
        (
            self.train_encoder_input,
            self.train_decoder_input,
            self.train_decoder_target,
            self.source_vocab,
            self.target_vocab,
        ) = self.train_data
        self.source_char2int, self.source_int2char = self.source_vocab
        self.target_char2int, self.target_int2char = self.target_vocab

        # create val data (only encode function suffices as the dictionary lookup should be kep the same.
        self.val_data = self.encode(
            self.val["src"].to_list(),
            self.val["tgt"].to_list(),
            list(self.source_char2int.keys()),
            list(self.target_char2int.keys()),
            source_char2int=self.source_char2int,
            target_char2int=self.target_char2int,
        )
        self.val_encoder_input, self.val_decoder_input, self.val_decoder_target = self.val_data
        self.source_char2int, self.source_int2char = self.source_vocab
        self.target_char2int, self.target_int2char = self.target_vocab

        # create test data
        self.test_data = self.encode(
            self.test["src"].to_list(),
            self.test["tgt"].to_list(),
            list(self.source_char2int.keys()),
            list(self.target_char2int.keys()),
            source_char2int=self.source_char2int,
            target_char2int=self.target_char2int,
        )
        self.test_encoder_input, self.test_decoder_input, self.test_decoder_target = self.test_data
        self.source_char2int, self.source_int2char = self.source_vocab
        self.target_char2int, self.target_int2char = self.target_vocab

    


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
            source_char2int, source_int2char = self.dictionary_lookup(source_chars)
            target_char2int, target_int2char = self.dictionary_lookup(target_chars)

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


    def preprocess(self, source , target):
        source_chars = set()
        target_chars = set()

        source = [str(x) for x in source]
        target = [str(x) for x in target]

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

        return self.encode(source_words, target_words, source_chars, target_chars)

