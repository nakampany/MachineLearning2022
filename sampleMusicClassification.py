-- Google Colaboratory --
# 適当なコマンドを実行してランタイムを起動しておく
print("Hello, world")

#Google Driveをマウントして共有ドライブで共有した「ClassMachineLearning/genres」が見れることを確認する
!ls "/content/drive/Shareddrives/ClassMachineLearning/genres"
-- Google Colaboratory --

--- python ---
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from matplotlib.pyplot import specgram

sample_rate, X = scipy.io.wavfile.read("/content/drive/Shareddrives/ClassMachineLearning/genres/blues/blues.00084.wav")
print(sample_rate, X.shape)
specgram(X, Fs=sample_rate, xextent=(0,30))
plt.show()
--- python ---


-- Google Colaboratory --
以下のファイルをアップロードする
ceps_based_classifier.py
ceps.py
data.tar.gz
fft_based_classifier.py
fft.py
utils.py
-- Google Colaboratory --

--- python ---
!mkdir charts
from fft import plot_specgrams
plot_specgrams()
--- python ---

-- Google Colaboratory --
!apt install sox
!sox --null -r 22050 sine_a.wav synth 02 sine 400
!sox --null -r 22050 sine_b.wav synth 02 sine 3000
!sox --combine mix --volume 1 sine_b.wav --volume 0.5 sine_a.wav sine_mix.wav
-- Google Colaboratory --


--- python ---
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from matplotlib.pyplot import specgram

sample_rate, X = scipy.io.wavfile.read("sine_a.wav")
print(sample_rate, X.shape)
specgram(X, Fs=sample_rate, xextent=(0,30))
plt.show()

sample_rate, X = scipy.io.wavfile.read("sine_b.wav")
print(sample_rate, X.shape)
specgram(X, Fs=sample_rate, xextent=(0,30))
plt.show()

sample_rate, X = scipy.io.wavfile.read("sine_mix.wav")
print(sample_rate, X.shape)
specgram(X, Fs=sample_rate, xextent=(0,30))
plt.show()
--- python ---


--- python ---
from fft import plot_wav_fft_demo
plot_wav_fft_demo()

from fft import plot_wav_fft
plot_wav_fft("/content/drive/Shareddrives/ClassMachineLearning/genres/classical/classical.00000.wav", desc="some sample song")

plot_wav_fft("/content/drive/Shareddrives/ClassMachineLearning/genres/jazz/jazz.00000.wav", desc="some sample song")
--- python ---

--- python ---
!mkdir data
from fft import create_fft
create_fft("/content/drive/Shareddrives/ClassMachineLearning/genres/classical/classical.00000.wav")

import sys
import os
import glob
for fn in glob.glob(os.path.join("/content/drive/Shareddrives/ClassMachineLearning/genres/classical", "*.wav")):
        create_fft(fn)
--- python ---

-- Google Colaboratory --
! tar xvfz data.tar.gz
-- Google Colaboratory --

--- python 古い　---
from sklearn.cross_validation import ShuffleSplit
rs =ShuffleSplit(4, n_iter=3, test_size=.25, random_state=0)
len(rs)
print rs
for train_index, test_index in rs:
    print "TRAIN:", train_index, "TEST:", test_index
--- python ---

--- python 新しい　---
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
y = np.array([1, 2, 1, 2, 1, 2])
from sklearn.model_selection import ShuffleSplit 
rs =ShuffleSplit(n_splits=3, test_size=.25, random_state=0)
rs.get_n_splits(X)

print(rs)

for train_index, test_index in rs.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

--- python ---

--- python ---
# GENRE_LIST = ["classical", "jazz", "country", "pop", "rock", "metal"]
from utils import GENRE_LIST
from fft import read_fft
X, y = read_fft(GENRE_LIST)
print(X.shape, y.shape)

def create_model():
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=200)

from fft_based_classifier import train_model
train_avg, test_avg, cms = train_model(create_model, X, y, "Log Reg FFT", plot=True)

cm_avg = np.mean(cms, axis=0)
cm_norm = cm_avg / np.sum(cm_avg, axis=0)

print(cm_norm)
--- python ---

--- python ---
plot_confusion_matrix(cm_norm, GENRE_LIST, "fft",
                     "Confusion matrix of a FFT based classifier")
--- python ---

--- python ---
def create_model():
    from sklearn.svm import SVC
    # clf = SVC(kernel='linear', C=1.0, gamma=10.0, random_state=0, probability=True)
    clf = SVC(kernel='rbf', C=1.0, gamma="scale", random_state=0, probability=True)
    # clf = SVC(kernel='rbf', C=1.0, gamma="auto", random_state=0, probability=True)

    return clf

from fft_based_classifier import train_model
train_avg, test_avg, cms = train_model(create_model, X, y, "Log Reg FFT", plot=True)

cm_avg = np.mean(cms, axis=0)
cm_norm = cm_avg / np.sum(cm_avg, axis=0)

print(cm_norm)
--- python ---

--- python ---
from utils import plot_confusion_matrix
plot_confusion_matrix(cm_norm, GENRE_LIST, "fft",
                          "Confusion matrix of an FFT based classifier")
--- python ---

--- Google Colaboratory ---
# !pip install scikits.talkbox
!pip install python_speech_features

from scipy import io
from scipy.io import wavfile
from python_speech_features import mfcc

!python ceps.py classical

!python ceps.py jazz
!python ceps.py country
!python ceps.py pop
!python ceps.py rock
!python ceps.py metal
--- Google Colaboratory ---

-- Google Colaboratory --
from ceps import read_ceps
X, y = read_ceps(GENRE_LIST)
print(X.shape)

def create_model():
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=200)

    return clf

from ceps_based_classifier import train_model
train_avg, test_avg, cms = train_model(
    create_model, X, y, "Log Reg CEPS", plot=True)

cm_avg = np.mean(cms, axis=0)
cm_norm = cm_avg / np.sum(cm_avg, axis=0)
print(cm_norm)

plot_confusion_matrix(cm_norm, GENRE_LIST, "ceps",
                     "Confusion matrix of a CEPS based classifier")

-- Google Colaboratory --