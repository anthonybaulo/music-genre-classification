import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import cm
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import model_from_json


# Save/load funcs may be skipped in favor of 
# keras.save() and load_model()
def save_model_to_json(model, fpath):
    model_json = model.to_json()
    with open(fpath, "w") as json_file:
        json_file.write(model_json)

        
def load_compile_model(json_path, weights_path, opt='adam'):
    with open(json_path, 'r') as json_file:
        loaded_json = json_file.read()
    loaded = model_from_json(loaded_json)
    loaded.load_weights(weights_path)
    loaded.compile(loss='categorical_crossentropy',
                   optimizer=opt,
                   metrics=['accuracy'])
    return loaded    


def get_ytrue_ypred_targets(model, test_datagen):
    y_pred = model.predict_generator(test_datagen)
    y_pred = np.argmax(y_pred, axis=1)

    y_true = test_datagen.test_labels
    y_true = np.argmax(y_true, axis=1)

    target_names = sorted(test_datagen.label_dict.keys())
    return y_true, y_pred, target_names


def save_summary_plots(history, fpath='plt.png', dpi=200):
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    ax1, ax2 = axes.flatten()
    
    # Accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Test'], loc='best')

    # Loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Test'], loc='best')
    
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(fpath, dpi=dpi)
    
    
def save_confusion_matrix(y_true, y_pred, target_names, fpath='cm.png', dpi=200):
    fig, ax = plt.subplots(figsize=(6,6))
    
    mat = confusion_matrix(y_true, y_pred)

    sns.heatmap(mat.T, square=True, annot=True, fmt='d', 
                cbar=True, cmap=cm.Reds,
                xticklabels=target_names,
                yticklabels=target_names,
                ax=ax)

    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.xlabel('Actual', size=15)
    plt.ylabel('Predicted', size=15)
    plt.savefig(fpath, dpi=dpi)