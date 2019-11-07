# Execute from the data/ directory
# 
# Creates file structure
# Converts mp3 files in the audio directory to numpy arrays
# 
# Example usage:
# $ python convert.py Rock Hip-Hop --meta ./fma_metadata/tracks.csv --audio ./fma_small --size small
 
import os
import argparse
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

# Create argument parser
parser = argparse.ArgumentParser(description='Convert mp3 files to numpy arrays.')
parser.add_argument('genres', metavar='G', nargs='+',
                    help='list of genre names separated by space. Must match csv format.')
parser.add_argument('--meta', dest='tracks', default='./fma_metadata/tracks.csv',
                    help='path to the metadata tracks.csv')
parser.add_argument('--audio', dest='audio', default='./fma_small',
                    help='path to top directory of audio files')
parser.add_argument('--size', dest='size', default='small',
                    help='size of FMA dataset')
args = parser.parse_args()

# Populate global variables
GENRES = args.genres
META_DIR = Path(args.tracks)
AUDIO_DIR = Path(args.audio)
SIZE = args.size
SPLITS = ['training', 'validation', 'test']


# Helper functions

# From https://github.com/mdeff/fma/blob/master/utils.py
def get_audio_path(audio_dir, track_id):
    """
    Return the path to the mp3 given the directory where the audio is stored
    and the track ID.
    
    Parameters
    ----------
    audio_dir : str or Path
        Path to top level audio directory
    track_id : int
        Track number without leading zeros

    Returns
    -------
    str
        Path to mp3 file

    Examples
    --------
    >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
    >>> get_audio_path(AUDIO_DIR, 2)
    '../data/fma_small/000/000002.mp3'
    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')


# Based on https://github.com/priya-dwivedi/Music_Genre_Classification/
def create_spectrogram(track_id):
    '''
    Create melspectrogram, normalized between 0-1

    Parameters
    ----------
    track_id : int
        Track number without leading zeros

    Returns
    -------
    np.array
        Transposed melspectrogram, normalized
    '''
    filename = get_audio_path(AUDIO_DIR, track_id)
    y, sr = librosa.load(filename)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512)
    spect = librosa.power_to_db(spect, ref=np.max)
    return (spect + 80) / 80


def get_metadata():
    '''Returns df of metadata for dataset SIZE (Global var)'''
    TRACKS = pd.read_csv(META_DIR, index_col=0, header=[0, 1])
    keep_cols = [('set', 'split'), ('set', 'subset'),('track', 'genre_top')]
    TRACKS = TRACKS[keep_cols]
    TRACKS = TRACKS[TRACKS[('set', 'subset')] == SIZE]
    TRACKS['track_id'] = TRACKS.index
    return TRACKS


def get_genre_df(df):
    '''Returns df with containing GENRES (Global var)'''
    genre_mask = df[('track', 'genre_top')].isin(GENRES)
    return df[genre_mask]


# Modified from https://github.com/priya-dwivedi/Music_Genre_Classification/
def create_arrays(df, verbose=True):
    '''
    Creates numpy arrays (melspectrograms) of tracks in dataframe.
    Saves array in directory corresponding to split/genre in dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe representing one size of the dataset
        (small, medium, large)
    verbose : bool
        Print status updates during conversion.

    Returns
    -------
    list
        shapes of all spectrograms before adjustments
    '''
    shapes = []
    total = len(df)
    count = 0
    start = datetime.now()
    
    for _, row in df.iterrows():
        # Skips records in case of errors
        try:
            count += 1

            # Get metadata
            track_id = int(row['track_id'])
            genre = str(row[('track', 'genre_top')])
            split = str(row[('set', 'split')])

            # Create spectrogram
            spect = create_spectrogram(track_id)

            # Store shape
            shapes.append([track_id, spect.shape[0], spect.shape[1]])

            # Adjust for shape differences
            spect = spect[:, :640]

            # Save to appropriate folder
            fname = './{}/{}/{:06d}.npy'.format(split, genre, track_id)
            np.save(fname, spect)

            if verbose:
                if count%100 == 0:
                    elapsed = datetime.now() - start
                    start = datetime.now()
                    print("Processed {} of {} in {} minutes"
                          .format(count, total, elapsed.seconds/60))
        except:
            if verbose:
                print("Couldn't process: {} of {} - track {}"
                      .format(count, total, track_id))
            continue

    return shapes


if __name__ == "__main__":
    # Create directory structure
    for split in SPLITS:
        for genre in GENRES:
            os.makedirs(f'{split}/{genre}', exist_ok=True)
    os.makedirs('shapes', exist_ok=True)
    
    # Get appropriate df
    metadata = get_metadata()
    df = get_genre_df(metadata)

    # Convert and store np arrays
    shapes = create_arrays(df)

    # Save shapes
    fname = './shapes/{}_shapes.npy'.format('_'.join(GENRES))
    np.save(fname, shapes)

