# Execute from the data/ directory
# 
# Creates file structure
# Converts mp3 files in the audio directory to numpy arrays
# 
# Example usage:
# $ python convert.py Rock Hip-Hop --meta ../data/fma_metadata/tracks.csv --audio ../data/fma_small --size small
 
import os
import argparse
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

def create_dirs(splits, genres):
    '''
    Creates directory structure expected by DataGenerator

    Parameters
    ----------
    splits : list
        Data splits (train, valid, test)
    genres : list
        Genres from CLI args
    '''
    for split in splits:
        for genre in genres:
            os.makedirs(f'../data/{split}/{genre}', exist_ok=True)
    os.makedirs('../data/shapes', exist_ok=True)

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
def create_spectrogram(audio_dir, track_id):
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
    filename = get_audio_path(audio_dir, track_id)
    y, sr = librosa.load(filename)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512)
    spect = librosa.power_to_db(spect, ref=np.max)
    return (spect + 80) / 80


def get_metadata(meta_dir, size):
    '''Returns df of metadata for dataset size'''
    TRACKS = pd.read_csv(meta_dir, index_col=0, header=[0, 1])
    keep_cols = [('set', 'split'), ('set', 'subset'),('track', 'genre_top')]
    TRACKS = TRACKS[keep_cols]
    TRACKS = TRACKS[TRACKS[('set', 'subset')] == size]
    TRACKS['track_id'] = TRACKS.index
    return TRACKS


def get_genre_df(df, genres):
    '''Returns df containing specified genres'''
    genre_mask = df[('track', 'genre_top')].isin(genres)
    return df[genre_mask]


# Modified from https://github.com/priya-dwivedi/Music_Genre_Classification/
def create_arrays(df, audio_dir, verbose=True):
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
            spect = create_spectrogram(audio_dir, track_id)

            # Store shape
            shapes.append([track_id, spect.shape[0], spect.shape[1]])

            # Check minimum shape
            if spect.shape[1] >= 640:
                # Adjust for shape differences
                spect = spect[:, :640]

                # Save to appropriate folder
                fname = '../data/{}/{}/{:06d}.npy'.format(split, genre, track_id)
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

def main(meta_dir, audio_dir, size, genres, splits):
    # Create directory structure
    create_dirs(splits, genres)

    # Get appropriate df
    metadata = get_metadata(meta_dir, size)
    df = get_genre_df(metadata, genres)

    # Convert and store np arrays
    shapes = create_arrays(df, audio_dir)

    # Save shapes
    fname = '../data/shapes/{}_shapes.npy'.format('_'.join(genres))
    np.save(fname, shapes)


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Convert mp3 files to numpy arrays.')
    parser.add_argument('genres', metavar='G', nargs='+',
                        help='list of genre names separated by space. Must match csv format.')
    parser.add_argument('--meta', dest='tracks', default='../data/fma_metadata/tracks.csv',
                        help='path to the metadata tracks.csv')
    parser.add_argument('--audio', dest='audio', default='../data/fma_small',
                        help='path to top directory of audio files')
    parser.add_argument('--size', dest='size', default='small',
                        help='size of FMA dataset')
    args = parser.parse_args()

    # Get arguments
    genres = args.genres
    meta_dir = Path(args.tracks)
    audio_dir = Path(args.audio)
    size = args.size
    splits = ['training', 'validation', 'test']
    
    # Run main
    main(meta_dir, audio_dir, size, genres, splits)

