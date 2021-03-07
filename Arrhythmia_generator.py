#!/usr/bin/env python
# coding: utf-8

# In[29]:

"import packages"
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
import keras
import PIL
import random
from PIL import Image
import pickle


# In[30]:

"""This code generates data for the CNN to use. This means that the data is only loaded into the CNN
in batches, saving all the data being loaded and saved for the duration of the code running. 
The data is read in from the Train Data folder containing 12,000 individual beats that have had a 
continuous wavelet transform applied."""

class DataGenerator(keras.utils.Sequence):
    """Generates data for keras"""
    def __init__(self, Beat_array_IDs, Label_array, dim, batch_size, n_channels, n_classes, shuffle):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.Label_array = Label_array
        self.Beat_array_IDs = Beat_array_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(Beat_array_IDs))
        
        
        
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.Label_array) / self.batch_size))
    
    

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        Beat_array_IDs_temp = [self.Beat_array_IDs[k] for k in indexes]

        # Find list of labels
        Labels_temp = [self.Label_array[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(Beat_array_IDs_temp, Labels_temp)

        return X, y
    
    
    def on_epoch_end(self):
        # This shuffles the Beat_List_IDs and the labels the same way
        # We know also need to shuffle the indexes in a similar way
        if self.shuffle == True:
            temp = list(zip(self.Beat_array_IDs, self.Label_array, self.indexes)) 
            random.shuffle(temp) 
            self.Beat_array_IDs, self.Label_array, self.indexes = zip(*temp)
            self.Beat_array_IDs = np.array(self.Beat_array_IDs)
            self.Label_array = np.array(self.Label_array)  
            
            
    def __data_generation(self, Beat_array_IDs_temp, Labels_temp):
        # Generates data containing batch_size samples # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size,self.n_classes), dtype=int)
        # Now we need to create a batch of images from train data folder
        # run this loop up until the batch size has been loaded
        for j in range(self.batch_size):
            
            # This gives us a sample name to use
            Beat_ID = Beat_array_IDs_temp[j]
            
            # Set the j'th value of Y as the label for the CWT image
            Y[j] = Labels_temp[j]

            # Load a file from the folder containing the data
            filename = ("Train Data/sample_{}.npy".format(Beat_ID))
            X[j] = np.load(filename, allow_pickle = True)
            #print(filename, Y[j])
        return X, Y

