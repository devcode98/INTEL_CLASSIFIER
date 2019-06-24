import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as py

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation


''' now we have imported all the libraries related to the creation of the neural network'''
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('database/seg_train',
                                                 target_size = (150, 150),
                                                 batch_size = 100,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('database/seg_test',
                                            target_size = (150, 150),
                                            batch_size = 100,
                                            class_mode = 'categorical')

''' since we have now been able to read all the images we will be creating out own neural network '''
cnn=Sequential()
cnn.add(Convolution2D(100,(4,4),input_shape=(150,150,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2) ))
cnn.add(Convolution2D(100,(2,2),activation='relu'))

cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Flatten())
cnn.add(Dense(100))

cnn.add(Activation('softmax'))
''' this is the final layer that we are creating '''
cnn.add(Dense(6))
cnn.add(Activation('softmax'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
''' our model is ready'''

cnn.fit_generator(training_set,samples_per_epoch = 8000,
                         nb_epoch = 25,validation_data = test_set,
                         nb_val_samples = 3000)


