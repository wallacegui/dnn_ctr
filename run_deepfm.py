import os
import sys
import time
import numpy as np
import pandas as pd
import keras
import math
from tqdm import tqdm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Reshape, Lambda, concatenate, dot, add
from keras.layers import Dropout, GaussianDropout, multiply, SpatialDropout1D, BatchNormalization, subtract
from keras.models import Model,load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.framework import graph_util
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from metrics.auc_callback import AUCCallback
import time
from keras.preprocessing.sequence import pad_sequences
from models.deepfm import DeepFM
from utils import max_min_scaler
from sklearn.preprocessing import KBinsDiscretizer

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]

batch_size = 512
epochs = 100

if __name__ == "__main__":
    train_file = 'data/adult/train.proced.data'
    valid_file = 'data/adult/valid.proced.data'

    ## 1.  train  valid data    
    train = pd.read_csv(train_file)
    y_train = train[LABEL_COLUMN].values
    valid = pd.read_csv(valid_file)
    y_valid = valid[LABEL_COLUMN].values
    train_shape,valid_shape = train.shape[0],valid.shape[0]
    df = pd.concat([train,valid])

    ## 2. discretizer nums feature
    discretizer = KBinsDiscretizer(n_bins=50,encode='ordinal',strategy='quantile')
    for feature in CONTINUOUS_COLUMNS:
        df[feature] = discretizer.fit_transform(df[feature].values.reshape(-1,1)).reshape(-1,)
    train = df[:train_shape]
    valid = df[-valid_shape:]

    ## 3. process train data
    model_train = {name: train[name]  for name in CONTINUOUS_COLUMNS+CATEGORICAL_COLUMNS}    
    model_valid = {name: valid[name]  for name in CONTINUOUS_COLUMNS+CATEGORICAL_COLUMNS}    

    ## 4. make model train valid 
    features_num_dict = {}
    for feature in CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS:
        features_num_dict[feature] = int(df[feature].max()+1)

    model = DeepFM(CONTINUOUS_COLUMNS+CATEGORICAL_COLUMNS,features_num_dict).model
    validation_data=(model_valid,y_valid)

    model.compile("adam", "binary_crossentropy",metrics=['acc'],)
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    model.fit(model_train, y_train,batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,validation_data=validation_data,callbacks=[early_stopping,AUCCallback(validation_data=validation_data)])

    model = DeepFM(CONTINUOUS_COLUMNS+CATEGORICAL_COLUMNS,features_num_dict,with_fm=False).model
    validation_data=(model_valid,y_valid)

    model.compile("adam", "binary_crossentropy",metrics=['acc'],)
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    model.fit(model_train, y_train,batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,validation_data=validation_data,callbacks=[early_stopping,AUCCallback(validation_data=validation_data)])
