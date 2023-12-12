import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2

class Attention(tf.keras.layers.Layer):
    def __init__(self,input_shape,**kwargs):
        super(Attention,self).__init__(**kwargs)
        self.build(input_shape)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='atten_w',shape=(input_shape[-1],1),initializer='random_normal',trainable=True)
        self.b=self.add_weight(name='atten_b',shape=(input_shape[1],1),initializer='zeros',trainable=True)        
        super(Attention, self).build(input_shape)
 
    def call(self,x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        e = K.squeeze(e, axis=-1)   
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context