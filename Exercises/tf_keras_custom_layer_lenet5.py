# -*- coding: utf-8 -*-
"""
#################################################################################
custom layers
from custom_layer import my
my_dense_layer(units, ativation)

#################################################################################   
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Lambda, Activation
from tensorflow.keras import regularizers
import numpy as np
from keras import backend as K
########################################################################################################
########################################################################################################
# Activate the following lines for GPU's usage, comment these lines if no GPU's are avilable

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
# # Currently, memory growth needs to be the same across GPUs
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# tf.config.experimental.set_visible_devices(gpus[2],'GPU')


##########################################################################################################################
############################################################################################################################
# class CustomConv2D(Layer):
#     def __init__(self, filters, **kwargs):
#         self.filters = filters
#         self.kernel_size = (3, 3)
#         super(CustomConv2D, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # only have a 3x3 kernel
#         shape = self.kernel_size + (input_shape[-1], self.filters)
#         self.kernel = self.add_weight(name='kernel', shape=shape,
#                                       initializer='glorot_uniform')
#         super(CustomConv2D, self).build(input_shape)

#     def call(self, x):
#         # duplicate rows 0 and 2
#         dup_rows = K.stack([self.kernel[0]]*2 + [self.kernel[1]] + [self.kernel[2]]*2, axis=0)
#         # duplicate cols 0 and 2
#         dup_cols = K.stack([dup_rows[:,0]]*2 + [dup_rows[:,1]] + [dup_rows[:,2]]*2, axis=1)
#         # having a 5x5 kernel now
#         return K.conv2d(x, dup_cols)

#     def compute_output_shape(self, input_shape):
#         return input_shape[:-1] + (self.filters,)

# from keras.engine import Layer, InputSpec
# from keras.layers import Flatten
# import tensorflow as tf


# class CustomMaxPooling(Layer):
#     def __init__(self, k=1, **kwargs):
#         super().__init__(**kwargs)
#         self.input_spec = InputSpec(ndim=3)
#         self.k = k

#     def compute_output_shape(self, input_shape):
#         return input_shape[0], (input_shape[2] * self.k)

#     def call(self, inputs):
#         # swap last two dimensions since top_k will be applied along the last dimension
#         shifted_input = tf.transpose(inputs, [0, 2, 1])

#         # extract top_k, returns two tensors [values, indices]
#         top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

#         # return flattened output
#         return Flatten()(top_k)

#     def get_config(self):
#         config = {'k': self.k}
#         base_config = super().get_config()
#         return {**base_config, **config}


class my_dense_layer(Layer):   
    def __init__(self, units, kernel_size, filters, strides, activation=None, name=None, **kwargs):
        self.units = units
        self.activation = activation
        # self.kernel_size = kernel_size
        # self.filters = filters
        # self.strides = strides

        super(my_dense_layer, self).__init__(name=name, **kwargs) 

        self.conv2d = tf.nn.conv2d(filters, kernel_size, strides=0, padding='same')
        self.maxpool = tf.nn.max_pool(strides)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units':self.units, 
            'activation':self.activation,
        })
        return config

    def build(self, input_shape): 
        # Define weight matrix and bias vector
        self.W = self.add_weight(shape=[int(input_shape[-1]),self.units],
                                 initializer='he_uniform',
                                 #regularizer=regularizers.l2(0.0005),
                                 trainable=True, name='w', dtype='float32')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True, name='b',  dtype='float32')

        super(my_dense_layer, self).build(input_shape) 
 
    def call(self, x):
        units = self.units

        x = self.conv2a(x)
        # x = tf.nn.relu(x)
        x = self.maxpool(x)


        x += input_tensor
        return tf.nn.relu(x)
        # Produce layer output

        y = tf.add(tf.matmul(x, self.W),self.b)
        if not self.activation == None:
            y = Activation(self.activation)(y)
        
        

        return y

########################################################################################################
########################################################################################################

#%% Load data 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#change type to float32
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

x_val = x_train[-5000:,:]
x_train = x_train[:-5000,:]

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

y_val = y_train[-5000:,:]
y_train = y_train[:-5000,:]

########################################################################################################
########################################################################################################
#%% Define sparseConnect Model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Parameters
n_epochs = 20

N_inx = np.shape(x_train)[1]
N_iny = np.shape(x_train)[2]
N_out = np.size(y_train,1) #10

x_ = Input(shape=(N_inx,N_iny,1))
x1 = my_dense_layer(units=0, strides=0, filters=20, activation='relu',kernel_size=(5,5))(x_)
x2 = my_dense_layer(units=0, filters=0, strides=(2,2))(x1)
x3 = my_dense_layer(units=0, strides=0, filters=50, activation='relu',kernel_size=(5,5))(x2)
x4 = my_dense_layer(strides=(2,2))(x3)
x5 = my_dense_layer(units=500,activation='relu')(x4)
y = my_dense_layer(units=10,activation='softmax')(x5)
model = Model(inputs=x_, outputs=y)

model.summary()
  
########################################################################################################
########################################################################################################

#define optimizer and learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#model compilation
model.compile(optimizer = optimizer,
             loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

#%% Start training
history = model.fit(x=x_train, y=y_train, batch_size=64, epochs=n_epochs, verbose=1,
          validation_data=(x_val,y_val))
    
#model evaluation   
test_results = model.evaluate(x_test, y_test, batch_size=64)
print('test accuracy:', test_results[1])
