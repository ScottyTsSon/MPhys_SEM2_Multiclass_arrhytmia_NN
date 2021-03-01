#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
import keras
import pickle
from keras.models import Sequential
import sys
sys.path.append('P:/')
from Arrhythmia_generator import DataGenerator
#from Arrhythmia_generator_aug_less_classes import DataGenerator_aug
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
#from random_eraser import get_random_eraser

# In[ ]:

# Load in  file indices and labels
with open('CNN_data_label_OHE.pkl', 'rb') as f:
    file_labels = pickle.load(f)
print(file_labels)
file_indices = []
for i in range(12000):
    file_indices.append(i)
   

# In[ ]:

# Parameters
params = {'dim': (128,128),
          'batch_size': 25,
          'n_classes': 6,
          'n_channels': 2,
          'shuffle': True}

params_val = {'dim': (128,128),
          'batch_size': 25,
          'n_classes': 6,
          'n_channels': 2,
          'shuffle': False}

params_test = {'dim': (128,128),
          'batch_size': 1,
          'n_classes': 6,
          'n_channels': 2,
          'shuffle': False}


# In[ ]:


# Implement a sequential network
def createModel():
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
    return model


# In[ ]:
f1_tot = np.zeros((6,))
# Splitting the data into test and train sets 
X_final = file_indices
Y_final = file_labels
X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, test_size=0.20)

# Edit this commented to select how much of the 2000 you want  to use 
"""X_train = X_train[:1000]
Y_train = Y_train[:1000]
X_test = X_test[:200]
Y_test = Y_test[:200]"""


train_generator = DataGenerator(X_train, Y_train, **params)
val_generator = DataGenerator(X_test, Y_test, **params_val)


model = createModel()

# Evaluate the initial model on the test data
print('=============================================')
print('TESTING UNTRAINED NETWORK: ')
result = model.evaluate(val_generator, verbose = True)
print(result)

print('UNTRAINED PREDICT: ')
predictions = model.predict(val_generator)
print(predictions[0])
print('Actual:')
print(Y_test[0])

print('=============================================')
print('TRAINING: ')
history = model.fit_generator(generator = train_generator,
                              epochs=40, validation_data=val_generator,
                              verbose=True)

print('TESTING TRAINED NETWORK: ')
result = model.evaluate(val_generator)
print('Results:')
print(result)

print('TRAINED PREDICT')
predictions = model.predict(val_generator)
print(predictions[0])
print('Actual:')
print(Y_test[0])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.figure()
plt.plot(history.history['accuracy'], 'cornflowerblue')
plt.plot(history.history['val_accuracy'], 'lightcoral')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('Accuracy plot.pdf', dpi = 1200)
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'], 'cornflowerblue')
plt.plot(history.history['val_loss'], 'lightcoral')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('Loss plot.pdf', dpi = 1200)

print('=============================================')
print('EVALUATING SCORES:')
scores_generator = DataGenerator(X_test, Y_test, **params_test)
flat_predictions = model.predict_generator(scores_generator,verbose = True)
print('PREDICTIONS:')
print(flat_predictions)
flat_labels = np.array(Y_test)

print('First 10 labels and predictions:')
for i in range(10):
    print(i,') Actual:', Y_test[i])
    print(i,') Predicted:', flat_predictions[i])

# Take largest prediction as the actual prediction
maxes = np.argmax(flat_predictions, axis = 1)
for c,i in enumerate(maxes):
    flat_predictions[c] = np.zeros((6,))
    flat_predictions[c][i] = 1
    
print('Flat predictions again: ', flat_predictions)
print(flat_predictions.shape)
print(flat_labels.shape)

print('=============================================')
conf_mat = metrics.multilabel_confusion_matrix(flat_labels, flat_predictions)
print('MULTI-LABEL CONFUSION MATRICES:')
print(conf_mat)

cat_true = []
for i in flat_labels:
    cat_true.append(np.argmax(i))
cat_predictions = []
for i in flat_predictions:
    cat_predictions.append(np.argmax(i))
confusion_matrix = metrics.multilabel_confusion_matrix(cat_true, cat_predictions)
print('CONFUSION MATRIX: ')
print(confusion_matrix)

print('=============================================')
print('F1 SCORES: ')
print(f1_score(flat_labels, flat_predictions, average = None))
f1 = f1_score(flat_labels, flat_predictions, average = None)
f1_tot = f1_tot + f1
print(f1_tot)
print(f1.shape)

print('ACCURACY: ')
acc = accuracy_score(flat_labels, flat_predictions)
print(acc)
print('=============================================')


# In[ ]:
import os
i = 0
pathnames = []
with os.scandir('Small Test Data') as root_dir:
    for path in root_dir:
        if path.is_file():
            pathnames.append(path.name)
pathnames.sort()
print(pathnames)

images = []
for pathname in pathnames:
    with open('Small Test Data/{}'.format(pathname), 'rb') as file:
        image = (np.load(file))
        images.append(image)
        print('Added ', pathname)
#plt.imshow(images[0][:,:,0])
images = np.array(images)
test_cat_true = []
for i in range(5):
    test_cat_true.append(5)
for i in range(5):
    test_cat_true.append(1)
for i in range(5):
    test_cat_true.append(0)
for i in range(5):
    test_cat_true.append(2)
for i in range(5):
    test_cat_true.append(3)
for i in range(5):
    test_cat_true.append(4)
    
test_preds = model.predict(images)
test_cat_preds = []
for i in test_preds:
    test_cat_preds.append(np.argmax(i))
print('Test True: ', test_cat_true)
print('Test preds: ', test_cat_preds)

model.save('HengguiCNN/CNN_less_classes_less_norm.h5')
with open('HengguiCNN/History_CNN_less_classes_less_norm.pkl', 'wb') as f:
    pickle.dump(history.history,f)



