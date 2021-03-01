#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
import keras
import PIL
import random
from PIL import Image
import pickle
import keras.preprocessing.image
from keras.preprocessing.image import ImageDataGenerator


# In[30]:


class DataGenerator_aug(keras.utils.Sequence):
    """Generates data for keras"""
    def __init__(self, Beat_array_IDs, Label_array, batch_size=25, dim=(64,64), n_channels=5,
                 n_classes=2, shuffle=True):
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
        datagen = ImageDataGenerator(zoom_range=(0.9,1.1),
                            width_shift_range=0.1,height_shift_range=0.1)
        # Now we need to create a batch of 5 beat sample arrays

                   
        for j in range(self.batch_size):
            
            # This gives us a sample name to use
            Beat_ID = Beat_array_IDs_temp[j]
            
            # Set the j'th value of Y as the label for the five beat array
            Y[j] = Labels_temp[j]

            # Load a 5 beat sample from the folder
            # First find the right filename by opening the pickle of samples and selecting the index
            # Then opening this from the whole data set
            filename = ("Segmented Data/Segments CWT/sample_{}.npy".format(Beat_ID))
            # Now load in a 5 beat sample and set as first element in input array
            
            temp1 = np.load(filename, allow_pickle = True)
            add = np.zeros((128,128,1))
            new = np.concatenate((temp1,add),axis=-1)
            ran_aug = datagen.random_transform(new,seed=None)
            X[j] = ran_aug[:,:,:2]
            
            
            
            
        # POTENTIALLY CHANGE THIS PART IF WE ARE NOT USING CATEGORICAL LABELS. But it might be good to keep this
        # as it's more general so can easily adapt if we want to try and identify different types of Cardiac 
        # arrhythmia's
#         return X, keras.utils.to_categorical(Y, num_classes=self.n_classes)
        return X, Y

