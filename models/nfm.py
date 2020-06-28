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
from keras.engine.topology import Layer
import numpy as np
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam

from keras import backend as K
import tensorflow as tf
from layers.MyMeanPool import MyMeanPool


class NFM:
    def __init__(self,sparse_features,features_num_dict,with_fm=True,k=10,hidden_layer=[128],optimizer=Adam(0.001)):
        self.sparse_features = sparse_features
        self.features_num_dict = features_num_dict
        self.with_fm = with_fm
        self.k=k
        self.hidden_layer=hidden_layer
        self.optimizer = optimizer
        self.model = self.create_model()

    def create_model(self):
        inputs = []
        field_embeddings = []
        for feature in self.sparse_features:
            input = Input(shape=(1,),name=feature)
            inputs.append(input)
            field_embedding = Embedding(input_dim=self.features_num_dict[feature],output_dim=self.k)(input)
            field_embeddings.append(field_embedding)
        ## fm part 0.5 * [(vi*xi)2 - vi2 * xi2]
        keras_squere = Lambda(lambda x:K.square(x))
        vixi_squere = keras_squere(add(field_embeddings))
        vi_squere_xi_squere = add([keras_squere(x) for x in field_embeddings])
        y_fm = subtract([vixi_squere,vi_squere_xi_squere])
        y_fm = Reshape((self.k,))(y_fm)
        fc_input = BatchNormalization()(y_fm)
        ## fc part
        for n in self.hidden_layer:
            fc_input = Dense(n)(fc_input)
            fc_input = BatchNormalization()(fc_input)
            fc_input = Activation('relu')(fc_input)
        fc_input = Reshape((self.hidden_layer[-1],))(fc_input)
        ## lr part
        if self.with_fm:
            lr_input = concatenate([y_fm,fc_input],axis=-1)
        else:
            lr_input = fc_input
        lr_input = Dense(1)(lr_input)
        y_predict = Activation('sigmoid',name='y_predict')(lr_input)
        model = Model(inputs=inputs, outputs=y_predict)
        return model