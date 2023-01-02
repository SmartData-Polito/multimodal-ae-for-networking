# Multi-modal Autoencoders for Network Traffic Analysis
___

<div style="display:flex;">
  <img src=".logos/polito.png", width="200", style="margin-right:8px;">
  <img src=".logos/huawei.png", width="220", style="margin-left:8px;">
</div>


In this repository, we provide the source code and data used in the experiments 
of our paper "Multi-modal Autoencoders for Network Traffic Analysis", presented 
at the [Conference Name] conference in [Year]. The paper presents a multi-modal 
autoencoder (MAE) architecture for network traffic analysis, and evaluates it 
on three different traffic analysis tasks. 

The MAE architecture consists of multiple adaptation modules for handling 
different input measurements, and an integration module for creating intermediate 
embeddings. We demonstrate the benefits of this approach by implementing it on 
three different traffic analysis tasks and comparing it to alternative deep 
learning solutions. Our results show that the MAE performs on par or better 
than these alternatives, and that the representation it generates is rich and 
compact.

This repo includes an implementation of the multi-modal autoencoder (MAE) 
architecture proposed in the paper, using Python 3.7.9 and Keras. We also 
provide scripts for training the models and notebooks for running the 
experiments and reproducing the results reported in the paper.

We hope that this repository will be useful for researchers interested in 
network traffic analysis and multi-modal autoencoders. Please feel free to 
contact us if you have any questions or suggestions.

##  **Table of Content**
- [How to reproduce results in the paper?](#how-to-reproduce-results-in-the-paper)
- [Notebooks](#notebooks)
- [Scripts](#scripts)


## **Project Structure**

The repository is organized as follows:

- `notebooks`: This folder contains Jupyter notebooks for each of the three case studies, as well as notebooks for hyperparameter tuning and results aggregation and visualization.
- `training`: This folder contains scripts for training the MAE and deep classifiers for all case studies, as well as for the grid search.

## **Requirements**

To run the code in this repository, you will need the following software and libraries:

- Python 3.6 or higher
- [INSERT REQUIRED PYTHON LIBRARIES HERE]



## **How to reproduce results in the paper?**
___

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

- MIRAGE: a dataset of network traffic flows for mobile app classification, with 44,000 samples and 16 classes.
- DARKNET: a dataset of IP addresses for classifying groups based on coordinates, with 14,000 samples and 13 + 1 classes.
- ISCX: a dataset of network flows for traffic type classification, with 609 samples and 5 classes.

We provide the datasets in preprocessed form, using TSTAT to extract both quantities and entities. The notebooks in this repository show how to extract these features from the raw data.

To download the datasets, run the following commands:
```
# Download MIRAGE dataset
wget [INSERT URL HERE] -O mirage.zip
unzip mirage.zip

# Download DARKNET dataset
wget [INSERT URL HERE] -O darknet.zip
unzip darknet.zip

# Download ISCX dataset
wget [INSERT URL HERE] -O iscx.zip
unzip iscx.zip
```

Once the datasets are downloaded, they will be unzipped and ready to use. You can then follow the instructions in the notebooks to preprocess the data and train the models.

## **Notebooks**
___

1. [**Features engineering**](./notebooks/00-characterization.ipynb)
    - Task01 - Quantities and entities extraction
    - Task02 - Quantities and entities extraction
    - Task03 - Quantities and entities extraction
    - Datasets characterization
2. [**Task01: Traffic application classification**](./notebooks/01-mobile-application.ipynb)
    - Training of Multi-modal Autoencoder
    - Training of Deep classifiers (Raw features and MAE)
    - Validation of deep classifiers
    - Embeddings neighborhood evaluation
    - Shallow learners
    - Unsupervised clustering
3. [**Task02: Darknet traffic classification**](./notebooks/02-darknet-traffic.ipynb)
    - Training of Multi-modal Autoencoder
    - Training of Deep classifiers (Raw features and MAE)
    - Validation of deep classifiers
    - Embeddings neighborhood evaluation
4. [**Task03: Traffic category classification**](./notebooks/03-traffic-category.ipynb)
    - Training of Multi-modal Autoencoder
    - Training of Deep classifiers (Raw features and MAE)
    - Validation of deep classifiers
    - Embeddings neighborhood evaluation
5. [**Grid Search**](./notebooks/04-grid-search.ipynb)
    - Training of Multi-modal Autoencoders
    - Training of Deep classifiers
    - Validation of deep classifiers
6. [**Conclusions**](./notebooks/05-visualization.ipynb)
    - Experiments finalization
    - Data visualization


## **Training the Models**
___

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
    - [x] Shallow learners
    - [x] Clustering
    - [x] Grid Search
    - [ ] Finalization and visualization
    - [ ] Features engineering -- Starting datasets
- Scripts
    - [ ] Launcher
    - [ ] Single scripts
- Documentation
    - [ ] Docstring functions
    - [ ] README
    - [ ] Raw data folder on cluster