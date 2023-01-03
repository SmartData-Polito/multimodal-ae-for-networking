#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Luca Gioacchini

""" 
This script is a launcher for the models training. User can specify the task
for which the model should be trained, the number of epochs, batch size and 
the number of the fold (stratified k fold). It allows to train the models
for all the folds or only for the specified one.

Usage
-----
python run_tasks.py -t TASK_NUMBER \
                    -m MODEL_TYPE \
                    -e NUM_EPOCHS \
                    -b BATCH_SIZE \
                    -f FOLD_NUMBER

Examples
--------

To run the classifiers model for task 02 with 15 epochs and a batch size of 256 
for fold 3:
```
$ python train_tasks.py -t 02 -m classifiers -e 15 -b 256 -f 3
```

To run the MAE model for task 01 with 20 epochs and a batch size of 128 for all
5 folds:
```
$ python train_tasks.py -t 01 -m mae -e 20 -b 128
```

To run both the MAE and classifiers models for task 03 with 10 epochs and a 
batch size of 512 for all 5 folds:
```
$ python train_tasks.py -t 03 -e 10 -b 512
```

"""

import argparse
import subprocess
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# parse command line arguments
parser = argparse.ArgumentParser(description='Run task scripts')
parser.add_argument(
    '-t', '--task', default='01', help='task number (01, 02, or 03)')
parser.add_argument(
    '-m', '--model', default='all', help='model type (mae, classifiers, or all)')
parser.add_argument(
    '-e', '--epochs', default=20, type=int, help='number of epochs')
parser.add_argument(
    '-b', '--batch_size', default=128, type=int, help='batch size')
parser.add_argument(
    '-f', '--fold_number', default='all', help='fold number (0-4 or all)')

args = parser.parse_args()

if len(sys.argv) < 7:
    parser.print_help()
    exit()

# run the appropriate scripts
if args.model in ['mae', 'all']:
    if args.fold_number == 'all':
        for i in range(5):
            subprocess.run(['python', f'task{args.task}_mae.py', 
                            '--epochs', str(args.epochs), 
                            '--batch_size', str(args.batch_size), 
                            '--fold_number', str(i)])
    else:
        subprocess.run(['python', f'task{args.task}_mae.py', 
                        '--epochs', str(args.epochs), 
                        '--batch_size', str(args.batch_size), 
                        '--fold_number', str(args.fold_number)])

if args.model in ['classifiers', 'all']:
    if args.fold_number == 'all':
        for i in range(5):
            subprocess.run(['python', f'task{args.task}_classifiers.py', 
                            '--epochs', str(args.epochs), 
                            '--batch_size', str(args.batch_size), 
                            '--fold_number', str(i)])
    else:
        subprocess.run(['python', f'task{args.task}_classifiers.py', 
                        '--epochs', str(args.epochs), 
                        '--batch_size', str(args.batch_size), 
                        '--fold_number', str(args.fold_number)])