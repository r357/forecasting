import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from DataProcessing.DataProcessing import PrepData
from transformer.time2vec import *
from transformer.transformer import *


# Hyperparameters
seq_len = 128 
batch_size = 32

d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256


# data
X_train, Y_train, X_val, Y_val, X_test, Y_test = PrepData("BTC-USD", seq_len)


# model
def create_model():
  ''' init time and transformer layers '''
  time_embedding = Time2Vector(seq_len)
  attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    
  ''' construct model '''
  in_seq = Input(shape=(seq_len, 5))
  x = time_embedding(in_seq)
  x = Concatenate(axis=-1)([in_seq, x])
  x = attn_layer1((x, x, x))
  x = attn_layer2((x, x, x))
  x = attn_layer3((x, x, x))
  x = GlobalAveragePooling1D(data_format='channels_first')(x)
  x = Dropout(0.1)(x)
  x = Dense(64, activation='relu')(x)
  x = Dropout(0.1)(x)
  out = Dense(1, activation='linear')(x)

  model = Model(inputs=in_seq, outputs=out)
  model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
  return model



model = create_model()
model.summary()

callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding.hdf5', 
                                              monitor='val_loss', 
                                              save_best_only=True, 
                                              verbose=1)

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=35,
                    callbacks=[callbacks],
                    validation_data=(X_val, Y_val))

model = tf.keras.models.load_model()





# CHECK THIS
# https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/output/attention-is-all-you-need.png



