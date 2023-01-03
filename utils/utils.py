import numpy as np
from keras import layers
import warnings
from sklearn.exceptions import DataConversionWarning, UndefinedMetricWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

def get_datasets(kfolds, K, df):
    train_ips, val_ips, y_train, y_val = kfolds[K]
    X_train = df.loc[train_ips].drop(columns=['label'])
    X_val = df.loc[val_ips].drop(columns=['label'])
    
    X_train, X_val = X_train.to_numpy(), X_val.to_numpy()
    y_train, y_val = np.ravel(y_train), np.ravel(y_val)

    return X_train, X_val, y_train, y_val