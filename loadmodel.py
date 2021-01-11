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


model = tf.keras.models.load_model('./models/Transformer+TimeEmbedding.hdf5',
                                   custom_objects={'Time2Vector':Time2Vector,
                                                   'SingleAttention': SingleAttention,
                                                   'MultiAttention': MultiAttention,
                                                   'TransformerEncoder' :TransformerEncoder})
model.summary()
tf.keras.utils.plot_model(
  model,
  to_file='./modelplot.png',
  show_shapes=True, show_layer_names=True)




# CHECK THIS
# https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/output/attention-is-all-you-need.png



