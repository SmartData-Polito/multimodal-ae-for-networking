#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Luca Gioacchini

""" 
This script trains a multi-modal autoencoder for traffic application 
classification. It loads the features and stratified-k-folds order, manages the 
training and validation datasets, builds the model, trains it, and retrieve the
embeddings transforming the whole dataset. 
The trained model and the embeddings are saved to disk. The script takes as 
input the number of epochs, batch size, and fold number to use for training and
validation.

Usage example
-------------

$ python task01_mae.py --epochs 20 --batch-size 128 --K 3

"""

import sys
sys.path.append('../') # Make mltoolbox and utils reachable

import argparse
from mltoolbox.representation import MultimodalAE
from utils import get_datasets
from keras import layers
import pandas as pd
import numpy as np
import joblib


# Features path
FEATURES = '../data/task01/features'


def train_multimodal_autoencoder(epochs: int, batch_size: int, K: int):
    """
    Train a multi-modal autoencoder for traffic application classification.

    Parameters
    ----------
    epochs : int
        The number of epochs to use for training.
    batch_size : int
        The batch size to use for training.
    K : int
        The fold number to use for training and validation.

    Returns
    -------
    None
        The function saves the trained model and evaluation metrics to disk.

    """
    #=============================================================
    # Load features and stratified-k-folds order
    #=============================================================

    # Load ports word2vec embeddings - entity
    payload=pd.read_csv(f'{FEATURES}/payload.csv', index_col=[0])
    # Load statistics features - quantity
    statistics=pd.read_csv(f'{FEATURES}/statistics.csv', index_col=[0])
    # Load statistics sequences - quantity
    sequences=pd.read_csv(f'{FEATURES}/sequences.csv', index_col=[0])
    # Load ip address word2vec embeddings - entity
    ipaddress=pd.read_csv(f'{FEATURES}/ipaddress.csv', index_col=[0])
    # Merge the features as raw concatenation
    concat = payload.reset_index().drop(columns=['label'])\
                    .merge(statistics.reset_index().drop(columns=['label']), 
                           on='index', how='inner')\
                    .merge(sequences.reset_index().drop(columns=['label']), 
                           on='index', how='inner')\
                    .merge(ipaddress.reset_index(), on='index', how='inner')\
                    .set_index('index')
    # Load stratified k folds
    kfolds = joblib.load(f'../data/task01/skfolds/folds.save')

    #=============================================================
    # Manage training and validation dataset
    #=============================================================

    feature, fname = concat, 'mae'
    # Get the features size. Last column is the label one
    P,St,Se,I = (payload.shape[1]-1, 
                 statistics.shape[1]-1,
                 sequences.shape[1]-1, 
                 ipaddress.shape[1]-1)

    # Retrieve the training and validation samples from the k-folds order
    X_train, X_val, y_train, y_val = get_datasets(kfolds, K, feature)

    #=============================================================
    # Build the model
    #=============================================================

    # Define the classifier architecture
    inputs = layers.Input(X_train.shape[1],)

    # Encoder branch of modality 1 - payload embeddings
    hidden1 = layers.Lambda(lambda x: x[:, :P])(inputs)
    hidden1 = layers.Reshape((-1, 1))(hidden1)
    hidden1 = layers.Dense(64, activation='relu')(hidden1)
    hidden1 = layers.Conv1D(32, 3, activation='relu')(hidden1)
    hidden1 = layers.Dropout(.3)(hidden1)
    hidden1 = layers.MaxPooling1D(2)(hidden1)
    hidden1 = layers.Dropout(.3)(hidden1)
    hidden1 = layers.Flatten()(hidden1)
    hidden1 = layers.Dense(32, activation='relu')(hidden1)

    # Encoder branch of modality 2 - statistics
    hidden2 = layers.Lambda(lambda x: x[:, P:St])(inputs)
    hidden2 = layers.Dense(32, activation='relu')(hidden2)

    # Encoder branch of modality 3 - sequences
    hidden3 = layers.Lambda(lambda x: x[:, P+St:P+St+Se])(inputs)
    hidden3 = layers.Reshape((32, 4))(hidden3)
    hidden3 = layers.Conv1D(32, 3, activation='relu')(hidden3)
    hidden3 = layers.Dropout(.3)(hidden3)
    hidden3 = layers.MaxPooling1D(2)(hidden3)
    hidden3 = layers.Dropout(.3)(hidden3)
    hidden3 = layers.Flatten()(hidden3)
    hidden3 = layers.Dense(32, activation='relu')(hidden3)

    # Encoder branch of modality 4 - ip address embeddings
    hidden4 = layers.Lambda(lambda x: x[:, P+St+Se:P+St+Se+I])(inputs)
    hidden4 = layers.Dense(32, activation='relu')(hidden4)

    # Concatenate
    hidden = layers.Concatenate()([hidden1, hidden2, hidden3, hidden4])
    
    # Common encoder
    hidden = layers.Dense(512, activation='relu')(hidden)
    hidden = layers.Dense(256, activation='relu')(hidden)
    
    # Bottleneck
    hidden = layers.Dense(64, activation='relu', name='Coded')(hidden)
    
    # Common decoder
    hidden = layers.Dense(256, activation='relu')(hidden)
    hidden = layers.Dense(512, activation='relu')(hidden)
    hidden = layers.Dense(32*4, activation='relu')(hidden)

    # Decoder branch of modality 1 - payload embeddings
    hidden1 = layers.Dense(32, activation='relu')(hidden)
    hidden1 = layers.Dense(480, activation='relu')(hidden1)
    hidden1 = layers.Reshape((15, 32))(hidden1)
    hidden1 = layers.Conv1D(32, 3, strides=2, activation="relu", 
                            padding="same")(hidden1)
    hidden1 = layers.UpSampling1D(2)(hidden1)
    hidden1 = layers.Conv1D(4, 3, activation="relu", padding="same")(hidden1)
    hidden1 = layers.UpSampling1D(2)(hidden1)
    hidden1 = layers.Flatten()(hidden1)
    output1 = layers.Dense(P, activation='linear', name='payload')(hidden1)
    
    # Decoder branch of modality 2 - statistics
    hidden2 = layers.Dense(32, activation='relu')(hidden)
    output2 = layers.Dense(St, activation='linear', name='statistics')(hidden2)

    # Decoder branch of modality 3 - sequences
    hidden3 = layers.Dense(32, activation='relu')(hidden)
    hidden3 = layers.Dense(480, activation='relu')(hidden3)
    hidden3 = layers.Reshape((15, 32))(hidden3)
    hidden3 = layers.Conv1D(32, 3, strides=2, activation="relu", 
                            padding="same")(hidden3)
    hidden3 = layers.UpSampling1D(2)(hidden3)
    hidden3 = layers.Conv1D(4, 3, activation="relu", padding="same")(hidden3)
    hidden3 = layers.UpSampling1D(2)(hidden3)
    hidden3 = layers.Flatten()(hidden3)
    output3 = layers.Dense(Se, activation='linear', name='sequences')(hidden3)

    # Decoder branch of modality 4 - ip address embeddings
    hidden4 = layers.Dense(32, activation='relu')(hidden)
    output4 = layers.Dense(I, activation='linear', name='ipaddress')(hidden4)

    outputs = [output1, output2, output3, output4]

    # Mean Squared Errors
    loss = {'payload':'mse', 
            'statistics':'mse', 
            'sequences':'mse', 
            'ipaddress':'mse'} 
    # Balance losses
    weights = {'payload':1/P, 
               'statistics':1/St, 
               'sequences':1/Se, 
               'ipaddress':1/I}

    #=============================================================
    # Train the model
    #=============================================================

    # Initialize the classifier
    mae = MultimodalAE(model_path=f'../data/task01/mae/{fname}_k{K}',
                       io=(inputs, outputs), losses=loss, weights=weights)
    # Fit the multi-modal autoencoder
    mae.fit(training_data=(X_train, X_train), 
            y_sizes=[P, St, Se, I], 
            batch_size=batch_size, 
            scale_data=True, 
            epochs=epochs, 
            validation_data=(X_val, X_val), 
            save=True, 
            verbose=1)

    #=============================================================
    # Save the embeddings
    #=============================================================

    # Transform the dataset and save the embeddings
    embeddings = np.vstack([mae.transform(X_train), mae.transform(X_val)])
    # Manage the dataframe
    embeddings = pd.DataFrame(embeddings, index=np.hstack(kfolds[K][:2]))
    embeddings[['label']] = np.hstack(kfolds[K][2:]).reshape(-1, 1)
    embeddings.to_csv(f'../data/task01/embeddings/mae_embeddings_k{K}.csv')



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True, 
                        help="Number of epochs to train the model")
    parser.add_argument("--batch_size", type=int, required=True, 
                        help="Batch size to use during training")
    parser.add_argument("--fold_number", type=int, required=True, 
                        help="Fold number to use for training")
    args = parser.parse_args()

    # Train the multimodal autoencoder
    train_multimodal_autoencoder(args.epochs, 
                                 args.batch_size, 
                                 args.fold_number)
