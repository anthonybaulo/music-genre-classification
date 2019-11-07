import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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



# TODO: Create function to save model plots
# Consider directory and informative filename
def show_summary_stats(history, savepath=None, dpi=200):
    # List all data in history
    print(history.history.keys())

    # Summarize history for accuracy
    plt.figure(figsize=(5,3))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    if savepath:
        plt.savefig(os.path.join(savepath, f''), 
                    dpi=dpi)
    plt.show()

    # Summarize history for loss
    plt.figure(figsize=(5,3))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    if savepath:
        plt.savefig(savepath, dpi=dpi)
    plt.show()