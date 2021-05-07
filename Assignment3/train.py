import numpy as np
import pandas as pd

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import RNN, LSTM, GRU, Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

from dataProcessor import DataProcessing
from modelClass import Translation

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

'''
config_defaults = {
    "cell_type": str(args.cell_type),
    "latent_dim": int(args.latent_dim),
    "hidden": int(args.hidden),
    "optimiser": str(args.optimiser),
    "num_encoders": int(args.num_encoders),
    "num_decoders": int(args.num_decoders),
    "dropout": float(args.dropout),
    "epochs": int(args.epochs),
    "batch_size": int(args.batch_size),
}
'''

def train():

    config_defaults = {
        "cell_type": "RNN",
        "latentDim": 256,
        "hidden": 128,
        "optimiser": "rmsprop",
        "numEncoders": 1,
        "numDecoders": 1,
        "dropout": 0.2,
        "epochs": 15,
        "batch_size": 64,
    }


    wandb.init(config=config_defaults)
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

    modelInit = Translation(config,srcChar2Int=dataBase.source_char2int, tgtChar2Int=dataBase.target_char2int)
    
    model = modelInit.build_configurable_model()
    
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
    wandb.finish()
    



if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    '''    
    sweep_config = {
        "name": "Bayesian Sweep without attention",
        "method": "bayes",
        "metric": {"name": "val_accuracy", "goal": "maximize"},
        "parameters": {
            
            "cell_type": {"values": ["RNN", "GRU", "LSTM"]},
            
            "latentDim": {"values": [256]},
            
            "hidden": {"values": [128, 64]},
            
            "optimiser": {"values": ["rmsprop", "adam"]},
            
            "numEncoders": {"values": [1, 2, 3]},
            
            "numDecoders": {"values": [1, 2, 3]},
            
            "dropout": {"values": [0.1, 0.2, 0.3]},
            
            "epochs": {"values": [5,10,15]},
            
            "batch_size": {"values": [32, 64]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="CS6910-Assignment-3", entity="rahulsundar")

    wandb.agent(sweep_id, train)
    
    '''
    train()
    
