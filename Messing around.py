#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:07:25 2021

@author: josephcullen
"""

import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
import keras
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import Input, Model, optimizers
from keras.utils import plot_model


#model = Sequential()
#model.add(Conv2D(32, kernel_size = 10, activation='relu', input_shape=(128, 128, 2)))
"""rows, cols = 100,15
input_shape = Input(shape=(rows, cols, 1))

tower_1 = Conv2D(20, (100, 5), padding='same', activation='relu')(input_shape)
tower_1 = MaxPooling2D((1, 11), strides=(1, 1), padding='same')(tower_1)

tower_2 = Conv2D(20, (100, 7), padding='same', activation='relu')(input_shape)
tower_2 = MaxPooling2D((1, 9), strides=(1, 1), padding='same')(tower_2)

tower_3 = Conv2D(20, (100, 10), padding='same', activation='relu')(input_shape)
tower_3 = MaxPooling2D((1, 6), strides=(1, 1), padding='same')(tower_3)

merged = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
merged = Flatten()(merged)

out = Dense(200, activation='relu')(merged)
#out = Dense(num_classes, activation='softmax')(out)

model = Model(input_shape, out)
plot_model(model,to_file="model.png")"""


model = Sequential()
model.add(Conv2D(32, kernel_size = 10, activation='relu', input_shape=(128, 128, 2))),
model.add(Conv2D(32, kernel_size = 10, activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5)) # Weight decay rate? 1E-6
model.add(Conv2D(32, kernel_size = 8, activation = 'relu'))
model.add(Conv2D(32, kernel_size = 4, activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5)),
model.add(Dense(6, activation='softmax')) #softmax?
sgd = optimizers.SGD(lr=0.001, decay=0.000001, momentum=0.8, nesterov=True)
model.compile(optimizer=sgd, loss ='categorical_crossentropy', metrics=['mse', 'mae', 'categorical_accuracy'])
plot_model(model,to_file ='Model2.png')

