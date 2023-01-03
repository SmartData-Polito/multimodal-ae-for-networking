#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Luca Gioacchini

""" 
This script is a launcher for the gridsearch models training. User can specify 
the number of epochs, batch size and the number of the fold (stratified k fold). 
It allows to train the models for all the folds or only for the specified one.

Usage
-----
python train_gridsearch.py -e NUM_EPOCHS \
                           -b BATCH_SIZE \
                           -f FOLD_NUMBER

Examples
--------

To run the gridsearch training 15 epochs and a batch size of 256 for fold 3:
```
$ python train_gridsearch.py -e 15 -b 256 -f 3
```
To run the gridsearch training 15 epochs and a batch size of 256 for all
5 folds:
```
$ python train_tasks.py -e 20 -b 128
```

"""

import argparse
import subprocess
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# parse command line arguments
parser = argparse.ArgumentParser(description='Run gridsearch training')
parser.add_argument(
    '-e', '--epochs', default=20, type=int, help='number of epochs')
parser.add_argument(
    '-b', '--batch_size', default=128, type=int, help='batch size')
parser.add_argument(
    '-f', '--fold_number', default='all', help='fold number (0-4 or all)')

args = parser.parse_args()

if len(sys.argv) < 3:
    parser.print_help()
    exit()

# run the appropriate scripts
if args.fold_number == 'all':
    for i in range(5):
        subprocess.run(['python', f'gridsearch.py', 
                        '--epochs', str(args.epochs), 
                        '--batch_size', str(args.batch_size), 
                        '--fold_number', str(i)])
else:
    subprocess.run(['python', f'gridsearch.py', 
                    '--epochs', str(args.epochs), 
                    '--batch_size', str(args.batch_size), 
                    '--fold_number', str(args.fold_number)])