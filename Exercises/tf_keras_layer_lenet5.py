import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
seed(1)
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

#%%load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
matrix = np.dot(x_train.shape[1], x_train.shape[2])

#change type to float32
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# normalize inputs from 0-255 to 0-1
x_train = x_train / 255 #check how this works
x_test = x_test / 255

x_val = x_train[-5000:,:] #what does this mean? randomly taking samples?
x_train = x_train[:-5000,:]

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_val = y_train[-5000:,:]
y_train = y_train[:-5000,:]

#%%define model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D

n_epochs = 10 #10 training epochs

# N_in = np.size(x_train)[1] #784 = 28 * 28
N_out = np.size(y_train,1) #10
# print (N_in)
# print(N_out)
x_ = Input(shape=(28,28,1))
x1 = Conv2D(filters=20, 
            kernel_size=(5,5), 
            activation='relu',)(x_)
x2 = MaxPool2D((2,2))(x1) #is this the pooling size or strides? what does it mean if strides=(2,2)
x3 = Conv2D(filters=50, 
            kernel_size=(5,5), 
            activation='relu',)(x2)
x4 = MaxPool2D((2,2))(x3)
x5 = Dense(500,activation='relu')(x4)
y = Dense(10,activation='softmax')(x5)
model = Model(inputs=x_, outputs=y)

model.summary()


#define optimizer and learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#model compilation
model.compile(optimizer = optimizer,
             loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


#%%start training
history = model.fit(x_train, y_train, batch_size=64, epochs=n_epochs, verbose=1,
          validation_data=(x_val,y_val))
    
#model evaluation
test_results = model.evaluate(x_test, y_test, batch_size=64)
print('test accuracy:', test_results[1])