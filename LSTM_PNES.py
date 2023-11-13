from __future__ import division
import mne
import numpy as np
import scipy.signal
from scipy.signal import butter, lfilter
from matplotlib import pyplot
import math
from scipy.fftpack import fft, ifft
from scipy import signal
from scipy.signal import butter, lfilter, iirfilter, filtfilt
from scipy.signal import hilbert
from biosppy.signals import tools
import pandas as pd
from matplotlib import rc

from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter,iirfilter
from scipy.signal import hilbert
from numpy import array
from numpy import hstack


def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


csv_reader = pd.read_csv('C:/Users/wxiong/PycharmProjects/EEGstudy/Probability_LSTM_Cz_EEGauto_ACT0128.csv',sep=',',header=None)
Raw_auto_EEG= csv_reader.values
Raw_auto_EEG_arr=[]
for item in Raw_auto_EEG:
    Raw_auto_EEG_arr.append(float(item))

csv_reader = pd.read_csv('C:/Users/wxiong/PycharmProjects/EEGstudy/Probability_LSTM_ch31_RRIvar_ACT0128.csv',sep=',',header=None)
Raw_var_RRI= csv_reader.values
Raw_var_RRI_arr=[]
for item in Raw_var_RRI:
    Raw_var_RRI_arr.append(float(item))

csv_reader = pd.read_csv('C:/Users/wxiong/PycharmProjects/EEGstudy/likelihood_LSTM_ACT0128.csv',sep=',',header=None)
Raw_Pro= csv_reader.values
Raw_Pro_arr=[]
for item in Raw_Pro:
    Raw_Pro_arr.append(float(item))

in_seq1 = array(Raw_auto_EEG_arr)
in_seq2 = array(Raw_var_RRI_arr)
out_seq = array(Raw_Pro_arr)

in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

dataset = hstack((in_seq1, in_seq2, out_seq))


