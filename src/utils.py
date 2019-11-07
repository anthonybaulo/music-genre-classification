import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.metrics import confusion_matrix

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence


# Inspired by https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(Sequence):
    '''Generates .npy files for Conv2D'''
    def __init__(self, data_dir, include=None, batch_size=32, 
                 dim=(128,640), n_channels=1, test=False):
        '''
        Parameters
        ----------
        data_dir : str
            Path to data split (training, validation, or test)
        include : list or None
            Subdirectories to include
            if None, include all
        batch_size : int
            Number of files to return at a time
            Auto set to 1 if test=True
        dim : tuple
            Dimension of arrays to read in
        n_channels : int
            Number of color channels for image array
        test : bool
            If test split, store labels, do not shuffle indices,
            and set batch_size to 1
        '''
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.test = test
        self.include = include
        self.test_labels = None
        
        self.label_dict = self.__get_label_dict()
        self.files = self.__get_files()
        self.n_classes = len(self.label_dict)   # Number of sub dirs
        self.on_epoch_end()                    # populates self.indexes
        if self.test:
            self.test_labels = np.empty((len(self.files), self.n_classes), dtype=int)
            self.batch_size = 1
            
    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        idxs = self.indexes[index*self.batch_size : (index+1)*self.batch_size]

        # Find list of IDs
        file_list = self.files[idxs]

        # Generate data
        X, y = self.__data_generation(file_list)
        
        if self.test:
            self.test_labels[idxs,] = y

        return X, y

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.files))
        if not self.test:
            np.random.shuffle(self.indexes) # Shuffles in place
            
    def __get_files(self):
        '''Get all files from subdirectories of data_dir'''
        subdirs = [k for k in self.label_dict.keys()]
        all_files = []

        for subdir in subdirs:
            full_dir = os.path.join(self.data_dir, subdir)
            files = os.listdir(full_dir)
            for file in files:
                all_files.append(os.path.join(subdir, file))

        return np.array(all_files)

    def __get_label_dict(self):
        '''
        Create dict of labels from sub directories
        {Genre : int}
        '''
        subdirs = sorted(os.listdir(self.data_dir))
        
        # Only include specific sub dirs
        if self.include:
            subdirs = [s for s in subdirs if s in self.include]
        
        labels = np.arange(len(subdirs))
        return {k:v for k,v in zip(subdirs, labels)}
    
    def __data_generation(self, file_list):
        '''
        Generates data containing batch_size samples
        
        Parameters
        ----------
        file_list : list or np.array
            List of files to retrieve/process/load
        
        Returns
        -------
        X : (n_samples, *dim, n_channels)
        
        '''  
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, file in enumerate(file_list):
            npy = np.load(os.path.join(self.data_dir, file))
            target = file.split('/')[0]
            label = self.label_dict[target]
            X[i,] = npy[:,:,None]   # Create extra dim for channel
            y[i,] = label

        return X, to_categorical(y, num_classes=self.n_classes, dtype='int')


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


def get_true_pred_targets(model, test_datagen):
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