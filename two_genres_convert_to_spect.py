# coding: utf-8

import os
import numpy as np
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

import warnings
warnings.filterwarnings('ignore')


META_DIR = Path('data/fma_metadata')
AUDIO_DIR = Path('data/fma_small')
CONVERTED_DIR = Path('data/converted/rock_inst')

# Helper funcitons
# From https://github.com/mdeff/fma/blob/master/utils.py

def get_audio_path(audio_dir, track_id):
    """
    Return the path to the mp3 given the directory where the audio is stored
    and the track ID.
    Examples
    --------
    >>> import utils
    >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
    >>> utils.get_audio_path(AUDIO_DIR, 2)
    '../data/fma_small/000/000002.mp3'
    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')


def get_tids_from_directory(audio_dir):
    """Get track IDs from the mp3s in a directory.
    Parameters
    ----------
    audio_dir : str
        Path to the directory where the audio files are stored.
    Returns
    -------
        A list of track IDs.
    """
    tids = []
    for _, dirnames, files in os.walk(audio_dir):
        if dirnames == []:
            tids.extend(int(file[:-4]) for file in files)
    return tids

# Based on https://github.com/priya-dwivedi/Music_Genre_Classification/

def create_spectrogram(track_id):
    filename = get_audio_path(AUDIO_DIR, track_id)
    y, sr = librosa.load(filename)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512)
    spect = librosa.power_to_db(spect, ref=np.max)
    return spect.T
    
def create_array(df, genre_dict):
    genres = []
    X_spect = np.empty((0, 640, 128))
    total = len(df)
    count = 0
    #Code skips records in case of errors
    for index, row in df.iterrows():
        try:
            count += 1
            track_id = int(row['track_id'])
            genre = str(row[('track', 'genre_top')])
            spect = create_spectrogram(track_id)

            # Normalize for small shape differences
            spect = spect[:640, :]
            X_spect = np.append(X_spect, [spect], axis=0)
            genres.append(genre_dict[genre])
            if count%100 == 0:
                print("Processed {} of {}"
                      .format(count, total))
        except:
            print("Couldn't process: {} of {} - track {}"
                  .format(count, total, track_id))
            continue
    y_arr = np.array(genres)
    return X_spect, y_arr

# Get metadata
tracks = pd.read_csv(META_DIR/'tracks.csv', index_col=0, header=[0, 1]) 

# Keep necessary columns
keep_cols = [('set', 'split'), ('set', 'subset'),('track', 'genre_top')]
df_all = tracks[keep_cols]

# Use small dataset
df_all = df_all[df_all[('set', 'subset')] == 'small']

# Move index to track_id column
df_all['track_id'] = df_all.index

## Trim down to 2 genres
rock = df_all[('track', 'genre_top')] == 'Rock'
inst = df_all[('track', 'genre_top')] == 'Instrumental'
mask = rock | inst
two_genre = df_all[mask]

genre_dict = {'Rock': 0, 'Instrumental': 1}

## Create train, validation and test subsets
df_train = two_genre[two_genre[('set', 'split')] == 'training'  ]
df_valid = two_genre[two_genre[('set', 'split')] == 'validation']
df_test  = two_genre[two_genre[('set', 'split')] == 'test'      ]

# Create/Shuffle Spectrogram Arrays
def shuffle(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]

def create_and_save(dfs, fnames, genre_dict):
    for df, fname in zip(dfs, fnames):
        X, y = create_array(df, genre_dict)
        X_shuf, y_shuf = shuffle(X, y)
        np.savez(CONVERTED_DIR/fname, X=X_shuf, y=y_shuf)


if __name__ == "__main__":
    dfs = [df_test, df_valid, df_train]
    fnames = ['test_arr', 'valid_arr', 'train_arr']

    create_and_save(dfs, fnames, genre_dict)






