from __future__ import division
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter,iirfilter
from scipy.signal import hilbert


def movingaverage(values, window_size):
    weights = (np.ones(window_size))/window_size
    a=np.ones(1)
    return lfilter(weights,a,values)


# def split_sequence(sequence, n_steps):
#     X, y = list(), list()
#     for i in range(len(sequence)):
#         end_ix = i + n_steps
#         if end_ix > len(sequence)-1:
#             break
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
#         X.append(seq_x)
#         y.append(seq_y)
#     return array(X), array(y)
# # define input sequence
# raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# n_steps = 3
# X, y = split_sequence(raw_seq, n_steps)
# # summarize the data
# # for i in range(len(X)):
# # 	print(X[i], y[i])
# # reshape from [samples, timesteps] into [samples, timesteps, features]
# n_features = 1
# X = X.reshape((X.shape[0], X.shape[1], n_features))
# print(X)
# model = Sequential()
# model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# model.fit(X, y, epochs=200, verbose=0)
# x_input = array([70, 80, 90])
# x_input = x_input.reshape((1, n_steps, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)
# ## Bidirectional LSTM for univariate time series forecasting
# from keras.layers import Bidirectional
# model = Sequential()
# model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# model.fit(X, y, epochs=200, verbose=0)
# x_input = array([70, 80, 90])
# x_input = x_input.reshape((1, n_steps, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)


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

# raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# n_steps_in, n_steps_out=3,2
# X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# # reshape from [samples, timesteps] into [samples, timesteps, features]
# n_features = 1
# X = X.reshape((X.shape[0], X.shape[1], n_features))
# print(X)
# model = Sequential()
# model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
# # model.add(LSTM(100, activation='relu'))
# model.add(Dense(n_steps_out))
# model.compile(optimizer='adam', loss='mse')
# model.fit(X, y, epochs=50, verbose=0)
# x_input = array([70, 80, 90])
# x_input = x_input.reshape((1, n_steps_in, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)



csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1230/Cz_EEGauto_QLD1230_15s_3h.csv',sep=',',header=None)
Raw_variance_EEG= csv_reader.values
Raw_var_EEG_arr=[]
for item in Raw_variance_EEG:
    Raw_var_EEG_arr.append(float(item))

# var_arr=[]
# for item in Raw_var_EEG_arr:
#     if item<1e-8:
#         var_arr.append(item)
#     else:
#         var_arr.append(var_arr[-1])
# Raw_var_EEG_arr=var_arr

var_arr=[]
for item in Raw_var_EEG_arr:
    if item<500:
        var_arr.append(item)
    else:
        var_arr.append(var_arr[-1])
Raw_var_EEG_arr=var_arr



fore_arr_EEGvars=[]
for k in range(26):
# for k in range(2):
    var_arr=Raw_var_EEG_arr[0:(18720+240*3*k)]
    long_rhythm_var_arr=movingaverage(var_arr,240*6)
    long_var_plot = long_rhythm_var_arr[(240*6+240*3*k):(18720+240*3*k)]
    # long_var_plot = long_rhythm_var_arr[(240 * 0 + 240 * 3 * k):(18720 + 240 * 3 * k)]
    var_trans = hilbert(long_var_plot)
    value_phase = np.angle(var_trans)
    rolmean_short_EEGvar=value_phase

    target_arr = []
    for i in range(432):
        target_arr.append(rolmean_short_EEGvar[i * 40])

    raw_seq = target_arr
    n_steps_in, n_steps_out = 24*6, 3*6
    X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model = Sequential()
    # model.add(LSTM(50, activation='relu', input_shape=(n_steps_in, n_features)))
    # model.add(LSTM(50, activation="relu", return_sequences=True, input_shape=(n_steps_in, n_features)))
    # model.add(LSTM(50, activation="relu" ))

    model.add(LSTM(50, activation="tanh",recurrent_activation="sigmoid", return_sequences=True,input_shape=(n_steps_in, n_features)))
    model.add(LSTM(50, activation="tanh",recurrent_activation="sigmoid",))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)

    x_input = array(rolmean_short_EEGvar[-24*6:])
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    fore_arr_EEGvars.append(yhat[0])
    print(yhat)
np.savetxt("phases_LSTM_Ch31_EEGauto_QLD1230.csv", fore_arr_EEGvars, delimiter=",", fmt='%s')