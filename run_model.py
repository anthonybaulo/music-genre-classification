import sys
import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.append('./src')

from utils import DataGenerator

import tensorflow as tf
from tensorflow.keras import backend, regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Dropout, \
                                    Activation, MaxPooling2D, Flatten, \
                                    BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical



def build_model():
    model = Sequential()
    model.add(Conv2D(32, (9, 9), 
                    input_shape = (128, 640, 1), 
                    padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))

    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dropout(0.3))

    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(0.3))

    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dropout(0.3))

    model.add(Dense(units = 3, activation = 'softmax'))

    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])




checkpoint_callback = ModelCheckpoint('../models/model2_with_datagen_best_val_loss.h5', 
                                      monitor='val_loss', mode='min',
                                      save_best_only=True, verbose=0)

reducelr_callback = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.8, 
                                      patience=2, min_delta=0.005, verbose=1)

callbacks_list = [checkpoint_callback, reducelr_callback]

history = model.fit_generator(generator=datagen, epochs=25,
                              validation_data=valid_datagen, verbose=1, 
                              callbacks=callbacks_list)