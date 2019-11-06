# Execute from the data/ directory
# 
# Example usage:
# $ python convert.py Rock Hip-Hop --meta ./fma_metadata/tracks.csv --audio ./fma_small --size small
 
import os
import argparse
import numpy as np
import librosa
import pandas as pd
from pathlib import Path

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
SPLITS = ['train', 'validation', 'test']

# Create appropriate Dataframe
TRACKS = pd.read_csv(META_DIR, index_col=0, header=[0, 1])
keep_cols = [('set', 'split'), ('set', 'subset'),('track', 'genre_top')]
TRACKS = TRACKS[keep_cols]
TRACKS = TRACKS[TRACKS[('set', 'subset')] == SIZE]
TRACKS['track_id'] = TRACKS.index

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
    Create transposed melspectrogram, normalized

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
    return (spect.T + 80) / 80


def get_df(genre, split, size=SIZE):
    pass


# Based on https://github.com/priya-dwivedi/Music_Genre_Classification/
def create_array(df, genre_dict):
    '''
    Creates numpy arrays (melspectrograms) of tracks in dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe representing one split of data 
        (train, test, or validation)

    Returns
    -------

    '''
    genres = []
    X_spect = np.empty((0, 640, 128))
    total = len(df)
    count = 0
    # Code skips records in case of errors
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


if __name__ == "__main__":
    # Create directory structure
    for split in SPLITS:
        for genre in GENRES:
            os.makedirs(f'{split}/{genre}', exist_ok=True)
    
