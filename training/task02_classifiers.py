#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Luca Gioacchini

""" 
This script trains a set of deep classifiers on different feature sets for 
darknet IP classification. It loads the features, stratified-k-folds 
order, and pre-trained multimodal embeddings, manages the training and 
validation datasets, then builds and trains the classifiers. The trained 
classifiers are saved to disk. The script takes as input the number of 
epochs, batch size, and fold number to use for training and validation.

Usage example
-------------

$ python task02_classifiers.py --epochs 20 --batch_size 128 --fold_number 3

"""

import sys
sys.path.append('../') # Make mltoolbox and utils reachable

import argparse
from mltoolbox.classification import DeepClassifier
from utils import get_datasets
from keras import layers
import pandas as pd
import joblib


# Features and embeddings paths
FEATURES = '../data/task02/features'
EMBEDDINGS = '../data/task02/embeddings'


def train_validate_classifiers(epochs: int, batch_size: int, K: int):
    """
    Train and validate a deep classifier on different features set. Namely it
    is trained on (i) raw features independently, (ii) raw concatenation and
    (iii) pre-trained multimodal embeddings.

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
    ports=pd.read_csv(f'{FEATURES}/ports.csv', index_col=[0])

    # Load statistics features - quantity
    statistics=pd.read_csv(f'{FEATURES}/statistics.csv', index_col=[0])

    # Load ip address word2vec embeddings - entity
    ipaddress=pd.read_csv(f'{FEATURES}/ipaddress.csv', index_col=[0])
    
    # Merge the features as raw concatenation
    concat = ports.reset_index().drop(columns=['label'])\
                  .merge(statistics.reset_index().drop(columns=['label']), 
                         on='src_ip', how='inner')\
                  .merge(ipaddress.reset_index(), 
                         on='src_ip', how='inner')\
                  .set_index('src_ip')
    
    # Load the pre-trained multimodal embeddings
    embeddings=pd.read_csv(f'{EMBEDDINGS}/mae_embeddings_k{K}.csv', 
                       index_col=[0])
    
    # Collect the features in a dictionary
    features = {'ports':ports,
                'statistics':statistics,
                'ipaddress':ipaddress,
                'rawcat':concat,
                'mae':embeddings}

    # Load stratified k folds
    kfolds = joblib.load(f'../data/task02/skfolds/folds.save')
    
    # Get the number of classes
    n_classes = ipaddress.value_counts('label').shape[0]

    #=============================================================
    # Train the classifiers
    #=============================================================

    for fname, feature in features.items():
        print(f'\nTraining {fname} deep classifier')
        
        # Retrieve the training and validation samples from the k-folds order
        X_train, X_val, y_train, y_val = get_datasets(kfolds, K, feature)
        
        #=============================================================
        # Training
        #=============================================================

        # Define the classifier architecture
        inputs  = layers.Input(X_train.shape[1],)
        hidden  = layers.Dense(512, activation='relu')(inputs)
        hidden  = layers.Dropout(.3)(hidden)
        hidden  = layers.Dense(256, activation='relu')(hidden)
        hidden  = layers.Dropout(.3)(hidden)
        outputs = layers.Dense(n_classes, activation='softmax')(hidden)
        
        # Initialize the classifier
        mpath = f'../data/task02/classifiers/{fname}_k{K}'
        classifier = DeepClassifier(io=(inputs, outputs), model_path=mpath)
        
        # Train the classifier. Standardize data before training
        classifier.fit(training_data=(X_train, y_train), 
                       validation_data=(X_val, y_val), 
                       scale_data=True, 
                       batch_size=batch_size, 
                       epochs=epochs, 
                       save=True)


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

    # Train and validate the classifiers
    train_validate_classifiers(args.epochs, 
                               args.batch_size, 
                               args.fold_number)
