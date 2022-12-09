# multi-modal-autoencoder

Introduction and description of the repo

##  **Table of Content**
- [How to reproduce results in the paper?](#how-to-reproduce-results-in-the-paper)
- [Notebooks](#notebooks)
- [Scripts](#scripts)


## **How to reproduce results in the paper? - TO CHANGE**
___

Note: This guide assumes a Debian-like system (tested on Ubuntu 20.04 & Debian 11).

1. Clone this repository
2. Open a terminal and run 
    ```
    $ ./setup.sh
    ```
    This will firstly download the datasets from https://url_tbd, then it will
    initialize a new virtual environment and download and install the python 
    library requirements. Finally, it will start a jupyter notebook
3. Run the notebooks. 
    Note that the `raw` data is used to create the intermediate datasets in the 
    `coNEXT` folder. Notebooks are provided (as Appendix) for this step. Given 
    the size of the raw traces a spark cluster is recommended for this step.

    Once the models and intermediate data are created in the `coNEXT` folder, 
    run the other notebooks that produce the results in the paper. For example, 
    to run the first notebook:
    ```
    $ jupyter-lab 01-darknet-overview.ipynb
    ```
    
    Note that each notebook is referred to a single fold of the stratified
    k fold cross validation.
4. When the notebook exploration is ended, remember to deactivate the virtual 
    environment:
    ```
   $  deactivate
    ```


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


## **Scripts**
___



## **Todo**
- Notebooks:
    - [x] Shallow learners
    - [x] Clustering
    - [ ] Grid Search
    - [ ] Finalization and visualization
    - [ ] Features engineering -- Starting datasets
- Scripts
    - [ ] Launcher
    - [ ] Single scripts
- Documentation
    - [ ] Docstring functions
    - [ ] README
    - [ ] Raw data folder on cluster