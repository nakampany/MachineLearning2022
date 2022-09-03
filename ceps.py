# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import os
import glob
import sys

import numpy as np
import scipy
import scipy.io.wavfile

from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version

if Version(sys.version) < "3.0":
    from scikits.talkbox.features import mfcc
else:
    from scipy import io
    from scipy.io import wavfile
    from python_speech_features import mfcc

from utils import GENRE_DIR, DATA_DIR


def write_ceps(ceps, fn):
    """
    Write the MFCC to separate files to speed up processing.
    """
    base_fn, ext = os.path.splitext(fn)
    # data_fn = base_fn + ".ceps"
    base_fn_without_ext = os.path.basename(base_fn)
    data_fn = DATA_DIR + "/"+ base_fn_without_ext + ".ceps"

    np.save(data_fn, ceps)
    print("Written", data_fn)


def create_ceps(fn):
    sample_rate, X = scipy.io.wavfile.read(fn)

    if Version(sys.version) < "3.0":
        ceps, mspec, spec = mfcc(X)
    else:
        ceps = mfcc(X)
    
    write_ceps(ceps, fn)
    


def read_ceps(genre_list, base_dir=DATA_DIR):
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(base_dir, genre + "*.ceps.npy")):
            print(fn)
            ceps = np.load(fn)
            num_ceps = len(ceps)
            X.append(
                np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
            y.append(label)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    glob_wav = os.path.join(GENRE_DIR+"/"+sys.argv[1], "*.wav")
    print(glob_wav)
    for fn in glob.glob(glob_wav):
        create_ceps(fn)
