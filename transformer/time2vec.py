import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


class Time2Vector(Layer):
  def __init__(self, seq_len, **kwargs):
    super(Time2Vector, self).__init__()
    self.seq_len = seq_len

  def build(self, input_shape):
    ''' init weights and biases with shape (batch, seq_len) '''
    self.weights_linear = self.add_weight(name="weight_linear",
                                          shape=(int(self.seq_len),),
                                          initializer="uniform",
                                          trainable=True)
    
    self.bias_linear = self.add_weight(name="bias_linear",
                                          shape=(int(self.seq_len),),
                                          initializer="uniform",
                                          trainable=True)
 
    self.weights_periodic = self.add_weight(name="weight_periodic",
                                          shape=(int(self.seq_len),),
                                          initializer="uniform",
                                          trainable=True)
    
    self.bias_periodic = self.add_weight(name="bias_periodic",
                                          shape=(int(self.seq_len),),
                                          initializer="uniform",
                                          trainable=True)
 

  def call(self, x):
    ''' calculate linear and periodic time features '''
    x = tf.math.reduce_mean(x[:,:,:4], axis=-1)
    time_linear = self.weights_linear * x *self.bias_linear
    time_linear = tf.expand_dims(time_linear, axis=-1)

    time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
    time_periodic = tf.expand_dims(time_periodic, axis=-1)
    return tf.concat([time_linear, time_periodic], axis=-1)


  def get_config(self):
    config = super().get_config().copy()
    config.update({'seq_len': self.seq_len})
    return config


