# Multi-modal Autoencoders for Network Traffic Analysis
___

In this repository, we provide the source code and data used in the experiments of our paper "Multi-modal Autoencoders for Network Traffic Analysis", presented at the [Conference Name] conference in [Year]. The paper presents a multi-modal autoencoder (MAE) architecture for network traffic analysis, and evaluates it on three different traffic analysis tasks. 

The MAE architecture consists of multiple adaptation modules for handling different input measurements, and an integration module for creating intermediate embeddings. We demonstrate the benefits of this approach by implementing it on three different traffic analysis tasks and comparing it to alternative deep learning solutions. Our results show that the MAE performs on par or better 
than these alternatives, and that the representation it generates is rich and compact.

This repo includes an implementation of the multi-modal autoencoder (MAE) architecture proposed in the paper, using Python 3.7.9 and Keras. We also provide scripts for training the models and notebooks for running the experiments and reproducing the results reported in the paper.

We hope that this repository will be useful for researchers interested in network traffic analysis and multi-modal autoencoders. Please feel free to contact us if you have any questions or suggestions.

##  **Table of Content**
- [Project structure](#project-structure)
- [Dependencies](#dependencies)
- [How to reproduce results in the paper?](#how-to-reproduce-results-in-the-paper)
- [Datasets](#datasets)
- [Notebooks](#notebooks)
- [Training the models](#training-the-models)
- [Citations](#citations)


## **Project structure**

The repository is organized as follows:

- `notebooks`: This folder contains Jupyter notebooks for each of the three case studies, as well as notebooks for hyperparameter tuning and results aggregation and visualization.
- `training`: This folder contains scripts for training the MAE and deep classifiers for all case studies, as well as for the grid search.

## **Dependencies**

To run the code in this repository, you will need the following software and libraries:

- Python 3.6 or higher
- gensim
- community



## **How to reproduce results in the paper?**


Note: This guide assumes a Debian-like system (tested on Ubuntu 20.04 & Debian 11).

1. Clone this repository
    ```
    $ git clone https://github.com/[USERNAME]/[REPOSITORY]
    ```
2. Download the datasets

3. Install the prerequisites. Open a terminal and run 
    ```
    $ ./setup.sh
    ```
    This will firstly download the datasets from `https://url_tbd`, then it will
    initialize a new virtual environment and download and install the python 
    library requirements. Finally, it will start a jupyter notebook

4. Run the notebooks. 
    To run the first notebook:
    ```
    $ jupyter-lab 01-darknet-overview.ipynb
    ```
    
    Note that each notebook is referred to a single fold of the stratified
    k fold cross validation.

5. When the notebook exploration is ended, remember to deactivate the virtual 
    environment:
    ```
   $  deactivate
    ```


## **Datasets**

The datasets used in the paper can be downloaded from [INSERT URL HERE] using the password provided. These datasets are:

- `MIRAGE`: a dataset of network traffic flows for mobile app classification, with 44,000 samples and 16 classes.
- `DARKNET`: a dataset of IP addresses for classifying groups based on coordinates, with 14,000 samples and 13 + 1 classes.
- `ISCXVPN2016`: a dataset of network flows for traffic type classification, with 609 samples and 5 classes.

We provide the datasets in preprocessed form, using Tstat to extract quantities and Word2Vec to extract entities from a corpus file in .txt format. 

Once the datasets are downloaded and unzipped, they are ready to use. You can then follow the instructions in the notebooks to preprocess the data and train the models.

## **Notebooks**


1. [**Features engineering**](./notebooks/00-characterization.ipynb)
    - Task01 - Quantities and entities extraction
    - Task02 - Quantities and entities extraction
    - Task03 - Quantities and entities extraction
    - Datasets characterization
2. [**Task01: Traffic application classification**](./notebooks/01-mobile-application.ipynb)
    - Validation of deep classifiers
    - Embeddings neighborhood evaluation
    - Shallow learners
    - Unsupervised clustering
3. [**Task02: Darknet traffic classification**](./notebooks/02-darknet-traffic.ipynb)
    - Validation of deep classifiers
    - Embeddings neighborhood evaluation
4. [**Task03: Traffic category classification**](./notebooks/03-traffic-category.ipynb)
    - Validation of deep classifiers
    - Embeddings neighborhood evaluation
5. [**Grid Search**](./notebooks/04-grid-search.ipynb)
    - Validation of deep classifiers
6. [**Conclusions**](./notebooks/05-visualization.ipynb)
    - Experiments finalization
    - Data visualization


## **Training the models**

We provide the scripts to train models for (i) different tasks and folds and (ii) MAE gridsearch on task01. 

### Regular training
To train the MAE and the classifiers for a single task, users should specify the task number, model type, number of epochs, batch size, and fold number as command line arguments.

To run the training, open a terminal, move to the `training` folder and run:
```
python train_tasks.py -t TASK_NUMBER -m MODEL_TYPE -e NUM_EPOCHS -b BATCH_SIZE -f FOLD_NUMBER
```

#### Arguments

- `TASK_NUMBER`: The task number to run (01, 02, or 03). Default: 01
- `MODEL_TYPE`: The model type to run (mae, classifiers, or all). Default: all
- `NUM_EPOCHS`: The number of epochs to run. Default: 20
- `BATCH_SIZE`: The batch size to use. Default: 128
- `FOLD_NUMBER`: The fold number to run (0-4 or all). Default: all

#### Examples

To run the classifiers model for task 02 with 15 epochs and a batch size of 256 for fold 3:

```bash
$ cd training
$ python train_tasks.py -t 02 -m classifiers -e 15 -b 256 -f 3
```

To run the MAE model for task 01 with 20 epochs and a batch size of 128 for all 5 folds:
```bash
$ cd training
$ python train_tasks.py -t 01 -m mae -e 20 -b 128
```

To run both the MAE and classifiers models for task 03 with 10 epochs and a batch size of 512 for all 5 folds:

```bash
$ cd training
$ python train_tasks.py -t 03 -e 10 -b 512
```

### GridSearch training
To train the models, open a terminal, move to the `training` folder and run:
```
python run_gridsearch.py -e NUM_EPOCHS -b BATCH_SIZE -f FOLD_NUMBER
```

#### Arguments

- `NUM_EPOCHS`: The number of epochs to run. Default: 20
- `BATCH_SIZE`: The batch size to use. Default: 128
- `FOLD_NUMBER`: The fold number to run (0-4 or all). Default: all

#### Examples

To run the gridsearch training 15 epochs and a batch size of 256 for fold 3:

```bash
$ cd training
$ python train_gridsearch.py -e 15 -b 256 -f 3
```

To run the gridsearch training 15 epochs and a batch size of 256 for all
5 folds:

```bash
$ cd training
$ python train_tasks.py -e 20 -b 128
```

## **Citations**
If you use this code or data in your research, please cite our paper:
```bibtex
@article{authors20xx,
  title={Multi-modal Autoencoders for Network Traffic Analysis},
  author={Authors, A. and Authors, B.},
  journal={Journal Name},
  year={20xx}
}
```


## **Todo**
- Notebooks:
    - [x] Task01
        - [x] Codes
        - [x] Comments
        - [x] Markdowns
    - [x] Task02
        - [x] Codes
        - [x] Comments
        - [x] Intro-Markdowns
    - [x] Task03
        - [x] Codes
        - [x] Comments
        - [x] Intro-Markdowns
    - [x] GridSearch
        - [x] Codes
        - [x] Comments
        - [x] Intro-Markdowns
    - [ ] Finalization and visualization
    - [ ] Starting datasets and characterization
        - [x] Datasets
        - [ ] Dataset balancing
        - [ ] Markdowns
        - [ ] Comments and functions
        - [ ] Stratified k fold
        - [ ] t-SNE - task01
- Scripts
    - [x] Training task01
    - [x] Training task02
    - [x] Training task03
    - [x] Training GridSearch
- Documentation
    - [x] README
    - [x] Raw data folder on cluster
    - [ ] References to the datasets and papers
    - [ ] More detailed instruction on how to download and setup experiments (once agreed)