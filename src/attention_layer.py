import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class TemporalAttention(Layer):
    """
    Custom Temporal Attention Layer.
    Computes a weighted sum over the time dimension of the input.
    Input shape: (batch_size, time_steps, features)
    Output shape: (batch_size, features)
    """
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = (batch, time_steps, features)
        self.W = self.add_weight(name="att_weight", 
                                 shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.b = self.add_weight(name="att_bias", 
                                 shape=(input_shape[1], 1),
                                 initializer="zeros", 
                                 trainable=True)
        super(TemporalAttention, self).build(input_shape)

    def call(self, x):
        # Alignment scores e = tanh(Wx + b)
        # x is (batch, 72, 8)
        # W is (8, 1)
        # Dot product results in (batch, 72, 1)
        e = K.tanh(K.dot(x, self.W) + self.b)
        
        # Remove last dimension to get (batch, 72)
        e = K.squeeze(e, axis=-1)
        
        # Compute weights alpha = softmax(e)
        alpha = K.softmax(e)
        
        # Add dimension back for weighting: (batch, 72, 1)
        alpha = K.expand_dims(alpha, axis=-1)
        
        # Context vector is the weighted sum: sum(alpha * x) 
        # Result shape: (batch, 8)
        context = x * alpha
        context = K.sum(context, axis=1)
        
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(TemporalAttention, self).get_config()
