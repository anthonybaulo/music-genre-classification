import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import classification_report

from src.datagen import DataGenerator
from src.utils import get_ytrue_ypred_targets, save_confusion_matrix, save_summary_plots

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Dropout, \
                                    Activation, MaxPooling2D, Flatten, \
                                    BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam



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

    model.add(Dense(units = 4, activation = 'softmax'))

    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model

def get_cb_list(name):
    earlystop_callback = EarlyStopping(monitor='val_loss', mode='min', 
                                       patience=20, min_delta=0.005, verbose=1,   
                                       restore_best_weights=True)
    
    checkpoint_callback = ModelCheckpoint(f'./models/{name}_weights_best_val_loss.h5', 
                                          monitor='val_loss', mode='min',
                                          save_best_only=True, verbose=1)

    reducelr_callback = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.85, 
                                          patience=2, min_delta=0.003, verbose=1)

    callbacks_list = [checkpoint_callback, reducelr_callback, earlystop_callback]

    return callbacks_list


def get_datagens(include=['Rock', 'Hip-Hop'], 
                 splits=['training', 'validation', 'test'], 
                 bs=[64,16,1]):
    datagens = []
    test = False

    for split, bs in zip(splits, bs):
        if 'test' in split:
            test = True

        datagen = DataGenerator('./data/'+split, include=include, 
                                batch_size=bs, dim=(128,640), 
                                n_channels=1, test=test)

        datagens.append(datagen)
    
    return tuple(datagens)


def main(name='model4'):
    # Build model
    model = build_model()

    # Get callbacks
    cbs = get_cb_list(name)

    # Get datagens
    train_dg, valid_dg, test_dg = get_datagens(include=['Rock', 'Hip-Hop', 'Instrumental', 'Folk'], 
                                               splits=['training', 'validation', 'test'], 
                                               bs=[64,16,1])

    # Train
    history = model.fit_generator(generator=train_dg, epochs=200,
                                  validation_data=valid_dg, verbose=0, 
                                  callbacks=cbs)

    # Save
    model.save(f'./models/{name}_arch_and_weights.h5')
    save_summary_plots(history, fpath=f'./images/{name}_summary.png')
    y_true, y_pred, target_names = get_ytrue_ypred_targets(model, test_dg)
    save_confusion_matrix(y_true, y_pred, target_names, fpath=f'./images/{name}_cm.png')
    
    # Print
    acc = model.evaluate_generator(test_dg)[1]
    print('Accuracy on test: {:.2f}%\n'.format(acc*100))
    print(classification_report(y_true, y_pred, target_names=target_names))

    
if __name__ == "__main__":
    main(name='model5')