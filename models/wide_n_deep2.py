from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import os
import time
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Reshape, Lambda, concatenate, dot, add, Masking
from keras.layers import Dropout, GaussianDropout, multiply, SpatialDropout1D, BatchNormalization, subtract
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import numpy as np
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf


class WideDeep:
    def __init__(self,num_features,sparse_features,features_num_dict,k=10,hidden_layer=[256,128,64],optimizer=Adam(0.001)):
        self.num_features = num_features
        self.sparse_features = sparse_features
        self.features_num_dict = features_num_dict
        self.k=k
        self.hidden_layer=hidden_layer
        self.optimizer = optimizer
        self.model = self.create_model()
        # self.model.summary()

    def create_model(self):
        inputs = []
        nums_input = []
        sparse_embedding = []
        ####dealing with num_features
        for feature in self.num_features:
            input = Input(shape=(1,),name=feature)
            inputs.append(input)
            reshape = Reshape((1,))(input)
            nums_input.append(reshape)
        ####dealing with sparse input
        for feature in self.sparse_features:
            input = Input(shape=(1,),name=feature)
            inputs.append(input)
            # fm embeddings
            embed = Embedding(self.features_num_dict[feature], self.k, input_length=1, trainable=True)(input)
            reshape = Reshape((self.k,))(embed)
            sparse_embedding.append(reshape)
        #######dnn layer##########
        fc_input_layer =  concatenate(nums_input+sparse_embedding, axis=-1)
        ####### FC layers #######
        for i in self.hidden_layer:
            fc_input_layer = Dense(i)(fc_input_layer)
            normed = BatchNormalization()(fc_input_layer)
            fc_input_layer = Activation('relu')(normed)
        ########linear layer##########
        lr_layer = Dense(1)(fc_input_layer)
        preds = Activation('sigmoid',name='y_predict')(lr_layer)
        model = Model(inputs=inputs, outputs=preds)
        return model