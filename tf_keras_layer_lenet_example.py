# -*- coding: utf-8 -*-
"""

"""


import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
seed(1)

########################################################################################################
########################################################################################################
# Activate the following lines for GPU's usage, comment these lines if no GPU's are avilable

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
# Currently, memory growth needs to be the same across GPUs
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.config.experimental.set_visible_devices(gpus[0],'GPU')


########################################################################################################
########################################################################################################

#%% Load data 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# flatten 28*28 images to a 784 vector for each image
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape((x_train.shape[0], num_pixels)).astype('float32')
x_test = x_test.reshape((x_test.shape[0], num_pixels)).astype('float32')

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
#%% Define Model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

n_epochs = 10

N_in = np.shape(x_train)[-1]
N_out = np.size(y_train,-1)

x_ = Input(shape=(N_in,))
x = Dense(300,activation='relu')(x_)
x = Dense(100,activation='relu')(x)
y = Dense(10,activation='softmax')(x)
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

