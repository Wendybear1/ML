from __future__ import division
from numpy import array
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter,iirfilter
import pywt
import pandas as pd
import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,f1_score,roc_auc_score,precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from scipy.signal import lfilter
from sklearn.preprocessing import StandardScaler,LabelEncoder
import tensorflow as tf

def movingaverage(values, window_size):
    weights = (np.ones(window_size))/window_size
    a=np.ones(1)
    return lfilter(weights,a,values)



# # def split_sequence(sequence, n_steps):
# #     X, y = list(), list()
# #     for i in range(len(sequence)):
# #         end_ix = i + n_steps
# #         if end_ix > len(sequence)-1:
# #             break
# #         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
# #         X.append(seq_x)
# #         y.append(seq_y)
# #     return array(X), array(y)
# # # define input sequence
# # raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# # n_steps = 3
# # X, y = split_sequence(raw_seq, n_steps)
# # # summarize the data
# # # for i in range(len(X)):
# # # 	print(X[i], y[i])
# # # reshape from [samples, timesteps] into [samples, timesteps, features]
# # n_features = 1
# # X = X.reshape((X.shape[0], X.shape[1], n_features))
# # print(X)
# # model = Sequential()
# # model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
# # model.add(Dense(1))
# # model.compile(optimizer='adam', loss='mse')
# # model.fit(X, y, epochs=200, verbose=0)
# # x_input = array([70, 80, 90])
# # x_input = x_input.reshape((1, n_steps, n_features))
# # yhat = model.predict(x_input, verbose=0)
# # print(yhat)
# # ## Bidirectional LSTM for univariate time series forecasting
# # from keras.layers import Bidirectional
# # model = Sequential()
# # model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
# # model.add(Dense(1))
# # model.compile(optimizer='adam', loss='mse')
# # model.fit(X, y, epochs=200, verbose=0)
# # x_input = array([70, 80, 90])
# # x_input = x_input.reshape((1, n_steps, n_features))
# # yhat = model.predict(x_input, verbose=0)
# # print(yhat)
#
#
# def split_sequence(sequence, n_steps_in, n_steps_out):
#     X, y = list(), list()
#     for i in range(len(sequence)):
#         end_ix = i + n_steps_in
#         out_end_ix = end_ix + n_steps_out
#         if out_end_ix > len(sequence):
#             break
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
#         X.append(seq_x)
#         y.append(seq_y)
#     return array(X), array(y)
#
# # raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# # n_steps_in, n_steps_out=3,2
# # X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# # # reshape from [samples, timesteps] into [samples, timesteps, features]
# # n_features = 1
# # X = X.reshape((X.shape[0], X.shape[1], n_features))
# # print(X)
# # model = Sequential()
# # model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
# # # model.add(LSTM(100, activation='relu'))
# # model.add(Dense(n_steps_out))
# # model.compile(optimizer='adam', loss='mse')
# # model.fit(X, y, epochs=50, verbose=0)
# # x_input = array([70, 80, 90])
# # x_input = x_input.reshape((1, n_steps_in, n_features))
# # yhat = model.predict(x_input, verbose=0)
# # print(yhat)
#
#
#
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1230/RRI_ch31_rawvariance_QLD1230_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# Raw_var_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_var_EEG_arr.append(float(item))
#
# # var_arr=[]
# # for item in Raw_var_EEG_arr:
# #     if item<1e-8:
# #         var_arr.append(item)
# #     else:
# #         var_arr.append(var_arr[-1])
# # Raw_var_EEG_arr=var_arr
#
# # var_arr=[]
# # for item in Raw_var_EEG_arr:
# #     if item<500:
# #         var_arr.append(item)
# #     else:
# #         var_arr.append(var_arr[-1])
# # Raw_var_EEG_arr=var_arr
#
#
#
# fore_arr_EEGvars=[]
# for k in range(26):
# # for k in range(2):
#     var_arr=Raw_var_EEG_arr[0:(18720+240*3*k)]
#     long_rhythm_var_arr=movingaverage(var_arr,240*6)
#     long_var_plot = long_rhythm_var_arr[(240*6+240*3*k):(18720+240*3*k)]
#     rolmean_short_EEGvar=long_var_plot
#
#
#     target_arr = []
#     for i in range(432):
#         target_arr.append(rolmean_short_EEGvar[i * 40])
#
#     raw_seq = target_arr
#     n_steps_in, n_steps_out = 24*6, 3*6
#     X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
#
#     n_features = 1
#     X = X.reshape((X.shape[0], X.shape[1], n_features))
#     model = Sequential()
#     # model.add(LSTM(50, activation='relu', input_shape=(n_steps_in, n_features)))
#
#     model.add(LSTM(50, activation="relu", return_sequences=True, input_shape=(n_steps_in, n_features)))
#     model.add(LSTM(50, activation="relu" ))
#
#     # model.add(LSTM(50, activation="tanh",recurrent_activation="sigmoid", return_sequences=True,input_shape=(n_steps_in, n_features)))
#     # model.add(LSTM(50, activation="tanh",recurrent_activation="sigmoid",))
#     model.add(Dense(n_steps_out))
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(X, y, epochs=200, verbose=0)
#
#     x_input = array(rolmean_short_EEGvar[-24*6:])
#     x_input = x_input.reshape((1, n_steps_in, n_features))
#     yhat = model.predict(x_input, verbose=0)
#     fore_arr_EEGvars.append(yhat[0])
#     print(yhat)
# np.savetxt("add_LSTM_Ch31_RRIvar_QLD1230.csv", fore_arr_EEGvars, delimiter=",", fmt='%s')



# channels = ['Fz', 'F3', 'F4', 'F7', 'F8', 'C4', 'C3', 'Cz','Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'O1','O2']
# import os
# cA5_mean = [];cD5_mean = [];cD4_mean = [];
# cD3_mean = [];cD2_mean = [];cD1_mean = [];
# cA5_std = [];cD5_std = [];cD4_std = [];
# cD3_std = [];cD2_std = [];cD1_std = [];
# cA5_max = [];cD5_max = [];cD4_max = [];
# cD3_max = [];cD2_max = [];cD1_max = [];
# cA5_min = [];cD5_min = [];cD4_min = [];
# cD3_min = [];cD2_min = [];cD1_min = [];
#
# directory =r'/Users/wxiong/PycharmProjects/ML/ES_EEG_ictal/'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
#         for m in range(int(len(dataset) / (256 * 60 * 2 ))):
#         # for m in range(1):
#             cA5_list = []; cD5_list = []; cD4_list = [];
#             cD3_list = []; cD2_list = []; cD1_list = []
#             for i in range(17):
#                 signal=dataset.loc[0+256*60*2*m:256*60*2+256*60*2*m, :][channels[i]]
#                 signal=signal[0:256]
#                 cA5, cD5, cD4, cD3, cD2, cD1 =pywt.wavedec(signal, 'coif1', level=5)
#                 cA5_list.append(cA5);cD5_list.append(cD5);cD4_list.append(cD4);
#                 cD3_list.append(cD3);cD2_list.append(cD2);cD1_list.append(cD1);
#
#             cA5_mean.append(np.mean(cA5_list));
#             cD5_mean.append(np.mean(cD5_list))
#             cD4_mean.append(np.mean(cD4_list))
#             cD3_mean.append(np.mean(cD3_list))
#             cD2_mean.append(np.mean(cD2_list))
#             cD1_mean.append(np.mean(cD1_list))
#
#             cA5_std.append(np.std(cA5_list));
#             cD5_std.append(np.std(cD5_list))
#             cD4_std.append(np.std(cD4_list))
#             cD3_std.append(np.std(cD3_list))
#             cD2_std.append(np.std(cD2_list))
#             cD1_std.append(np.std(cD1_list))
#
#             cA5_max.append(np.max(cA5_list));
#             cD5_max.append(np.max(cD5_list))
#             cD4_max.append(np.max(cD4_list))
#             cD3_max.append(np.max(cD3_list))
#             cD2_max.append(np.max(cD2_list))
#             cD1_max.append(np.max(cD1_list))
#
#             cA5_min.append(np.min(cA5_list));
#             cD5_min.append(np.min(cD5_list))
#             cD4_min.append(np.min(cD4_list))
#             cD3_min.append(np.min(cD3_list))
#             cD2_min.append(np.min(cD2_list))
#             cD1_min.append(np.min(cD1_list))
#
# print(cA5_mean);print(cA5_std);print(cA5_max);print(cA5_min)
# print(len(cA5_mean))
#
# df_ES = pd.DataFrame(list(zip(cA5_mean, cA5_std,cA5_max, cA5_min, cD5_mean, cD5_std,cD5_max, cD5_min,
#                            cD4_mean, cD4_std,cD4_max, cD4_min,cD3_mean, cD3_std,cD3_max, cD3_min,
#                            cD2_mean, cD2_std,cD2_max, cD2_min,cD1_mean, cD1_std,cD1_max, cD1_min)),
#                columns =['cA5_mean', 'cA5_std','cA5_max', 'cA5_min', 'cD5_mean', 'cD5_std','cD5_max', 'cD5_min',
#                            'cD4_mean', 'cD4_std','cD4_max', 'cD4_min','cD3_mean', 'cD3_std','cD3_max', 'cD3_min',
#                            'cD2_mean', 'cD2_std','cD2_max', 'cD2_min','cD1_mean', 'cD1_std','cD1_max', 'cD1_min'])
#
# tag_list=np.ones(len(cA5_mean))
# df_ES['type']=tag_list
#
#
# channels = ['Fz', 'F3', 'F4', 'F7', 'F8', 'C4', 'C3', 'Cz','Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'O1','O2']
# import os
# cA5_mean = [];cD5_mean = [];cD4_mean = [];
# cD3_mean = [];cD2_mean = [];cD1_mean = [];
# cA5_std = [];cD5_std = [];cD4_std = [];
# cD3_std = [];cD2_std = [];cD1_std = [];
# cA5_max = [];cD5_max = [];cD4_max = [];
# cD3_max = [];cD2_max = [];cD1_max = [];
# cA5_min = [];cD5_min = [];cD4_min = [];
# cD3_min = [];cD2_min = [];cD1_min = [];
#
# directory =r'/Users/wxiong/PycharmProjects/ML/PNES_EEG_ictal/'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
#         for m in range(int(len(dataset) / (256 * 60 * 2))):
#         # for m in range(1):
#             cA5_list = []; cD5_list = []; cD4_list = [];
#             cD3_list = []; cD2_list = []; cD1_list = []
#             for i in range(17):
#                 signal=dataset.loc[0+256*60*2*m:256*60*2+256*60*2*m, :][channels[i]]
#                 signal=signal[0:256]
#                 cA5, cD5, cD4, cD3, cD2, cD1 =pywt.wavedec(signal, 'coif1', level=5)
#                 cA5_list.append(cA5);cD5_list.append(cD5);cD4_list.append(cD4);
#                 cD3_list.append(cD3);cD2_list.append(cD2);cD1_list.append(cD1);
#
#             cA5_mean.append(np.mean(cA5_list));
#             cD5_mean.append(np.mean(cD5_list))
#             cD4_mean.append(np.mean(cD4_list))
#             cD3_mean.append(np.mean(cD3_list))
#             cD2_mean.append(np.mean(cD2_list))
#             cD1_mean.append(np.mean(cD1_list))
#
#             cA5_std.append(np.std(cA5_list));
#             cD5_std.append(np.std(cD5_list))
#             cD4_std.append(np.std(cD4_list))
#             cD3_std.append(np.std(cD3_list))
#             cD2_std.append(np.std(cD2_list))
#             cD1_std.append(np.std(cD1_list))
#
#             cA5_max.append(np.max(cA5_list));
#             cD5_max.append(np.max(cD5_list))
#             cD4_max.append(np.max(cD4_list))
#             cD3_max.append(np.max(cD3_list))
#             cD2_max.append(np.max(cD2_list))
#             cD1_max.append(np.max(cD1_list))
#
#             cA5_min.append(np.min(cA5_list));
#             cD5_min.append(np.min(cD5_list))
#             cD4_min.append(np.min(cD4_list))
#             cD3_min.append(np.min(cD3_list))
#             cD2_min.append(np.min(cD2_list))
#             cD1_min.append(np.min(cD1_list))
#
# print(cA5_mean);print(cA5_std);print(cA5_max);print(cA5_min)
# print(len(cA5_mean))
#
# df_PNES = pd.DataFrame(list(zip(cA5_mean, cA5_std,cA5_max, cA5_min, cD5_mean, cD5_std,cD5_max, cD5_min,
#                            cD4_mean, cD4_std,cD4_max, cD4_min,cD3_mean, cD3_std,cD3_max, cD3_min,
#                            cD2_mean, cD2_std,cD2_max, cD2_min,cD1_mean, cD1_std,cD1_max, cD1_min)),
#                columns =['cA5_mean', 'cA5_std','cA5_max', 'cA5_min', 'cD5_mean', 'cD5_std','cD5_max', 'cD5_min',
#                            'cD4_mean', 'cD4_std','cD4_max', 'cD4_min','cD3_mean', 'cD3_std','cD3_max', 'cD3_min',
#                            'cD2_mean', 'cD2_std','cD2_max', 'cD2_min','cD1_mean', 'cD1_std','cD1_max', 'cD1_min'])
#
# tag_list=np.zeros(len(cA5_mean))
# df_PNES['type']=tag_list
#
# data_sum=df_ES.append(df_PNES, ignore_index=True)
#
# data_sum.to_csv('C:/Users/wxiong/Documents/PHD/2022_ML/ES_PNES_coeff.csv')


# channels = ['Fz', 'F3', 'F4', 'F7', 'F8', 'C4', 'C3', 'Cz','Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'O1','O2']
# import os
# cA5_mean = [];cD5_mean = [];cD4_mean = [];
# cD3_mean = [];cD2_mean = [];cD1_mean = [];
# cA5_std = [];cD5_std = [];cD4_std = [];
# cD3_std = [];cD2_std = [];cD1_std = [];
# cA5_max = [];cD5_max = [];cD4_max = [];
# cD3_max = [];cD2_max = [];cD1_max = [];
# cA5_min = [];cD5_min = [];cD4_min = [];
# cD3_min = [];cD2_min = [];cD1_min = [];
#
# directory =r'/Users/wxiong/PycharmProjects/ML/ES_EEG_ictal/'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
#         for m in range(int(len(dataset) / (256 * 60 * 2 ))):
#         # for m in range(1):
#             cA5_list = []; cD5_list = []; cD4_list = [];
#             cD3_list = []; cD2_list = []; cD1_list = []
#             for i in range(2):
#                 signal=dataset.loc[0+256*60*2*m:256*60*2+256*60*2*m, :][channels[i]]
#                 signal=signal[0:256]
#                 cA5, cD5, cD4, cD3, cD2, cD1 =pywt.wavedec(signal, 'coif1', level=5)
#                 cA5_list.append(cA5);cD5_list.append(cD5);cD4_list.append(cD4);
#                 cD3_list.append(cD3);cD2_list.append(cD2);cD1_list.append(cD1);
#
#             cA5_mean = cA5_mean + list(np.mean(cA5_list, axis=0));
#             cD5_mean = cD5_mean + list(np.mean(cD5_list, axis=0));
#             cD4_mean = cD4_mean + list(np.mean(cD4_list, axis=0));
#             cD3_mean = cD3_mean + list(np.mean(cD3_list, axis=0));
#             cD2_mean = cD2_mean + list(np.mean(cD2_list, axis=0));
#             cD1_mean = cD1_mean + list(np.mean(cD1_list, axis=0));
#
#             cA5_max = cA5_max + list(np.max(cA5_list, axis=0));
#             cD5_max = cD5_max + list(np.max(cD5_list, axis=0));
#             cD4_max = cD4_max + list(np.max(cD4_list, axis=0));
#             cD3_max = cD3_max + list(np.max(cD3_list, axis=0));
#             cD2_max = cD2_max + list(np.max(cD2_list, axis=0));
#             cD1_max = cD1_max + list(np.max(cD1_list, axis=0));
#
#             cA5_min = cA5_min + list(np.min(cA5_list, axis=0));
#             cD5_min = cD5_min + list(np.min(cD5_list, axis=0));
#             cD4_min = cD4_min + list(np.min(cD4_list, axis=0));
#             cD3_min = cD3_min + list(np.min(cD3_list, axis=0));
#             cD2_min = cD2_min + list(np.min(cD2_list, axis=0));
#             cD1_min = cD1_min + list(np.min(cD1_list, axis=0));
#
#             cA5_std = cA5_std + list(np.std(cA5_list, axis=0));
#             cD5_std = cD5_std + list(np.std(cD5_list, axis=0));
#             cD4_std = cD4_std + list(np.std(cD4_list, axis=0));
#             cD3_std = cD3_std + list(np.std(cD3_list, axis=0));
#             cD2_std = cD2_std + list(np.std(cD2_list, axis=0));
#             cD1_std = cD1_std + list(np.std(cD1_list, axis=0));
#
#
# print(cA5_mean);print(cA5_std);print(cA5_max);print(cA5_min)
# print(len(cA5_mean))
#
# df_ES = pd.DataFrame(list(zip(cA5_mean, cA5_std,cA5_max, cA5_min, cD5_mean, cD5_std,cD5_max, cD5_min,
#                            cD4_mean, cD4_std,cD4_max, cD4_min,cD3_mean, cD3_std,cD3_max, cD3_min,
#                            cD2_mean, cD2_std,cD2_max, cD2_min,cD1_mean, cD1_std,cD1_max, cD1_min)),
#                columns =['cA5_mean', 'cA5_std','cA5_max', 'cA5_min', 'cD5_mean', 'cD5_std','cD5_max', 'cD5_min',
#                            'cD4_mean', 'cD4_std','cD4_max', 'cD4_min','cD3_mean', 'cD3_std','cD3_max', 'cD3_min',
#                            'cD2_mean', 'cD2_std','cD2_max', 'cD2_min','cD1_mean', 'cD1_std','cD1_max', 'cD1_min'])
#
# tag_list=np.ones(len(cA5_mean))
# df_ES['type']=tag_list
#
#
# channels = ['Fz', 'F3', 'F4', 'F7', 'F8', 'C4', 'C3', 'Cz','Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'O1','O2']
# import os
# cA5_mean = [];cD5_mean = [];cD4_mean = [];
# cD3_mean = [];cD2_mean = [];cD1_mean = [];
# cA5_std = [];cD5_std = [];cD4_std = [];
# cD3_std = [];cD2_std = [];cD1_std = [];
# cA5_max = [];cD5_max = [];cD4_max = [];
# cD3_max = [];cD2_max = [];cD1_max = [];
# cA5_min = [];cD5_min = [];cD4_min = [];
# cD3_min = [];cD2_min = [];cD1_min = [];
#
# directory =r'/Users/wxiong/PycharmProjects/ML/PNES_EEG_ictal/'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
#         for m in range(int(len(dataset) / (256 * 60 * 2))):
#         # for m in range(1):
#             cA5_list = []; cD5_list = []; cD4_list = [];
#             cD3_list = []; cD2_list = []; cD1_list = []
#             for i in range(17):
#                 signal=dataset.loc[0+256*60*2*m:256*60*2+256*60*2*m, :][channels[i]]
#                 signal=signal[0:256]
#                 cA5, cD5, cD4, cD3, cD2, cD1 =pywt.wavedec(signal, 'coif1', level=5)
#                 cA5_list.append(cA5);cD5_list.append(cD5);cD4_list.append(cD4);
#                 cD3_list.append(cD3);cD2_list.append(cD2);cD1_list.append(cD1);
#
#
#             cA5_mean = cA5_mean + list(np.mean(cA5_list, axis=0));
#             cD5_mean = cD5_mean + list(np.mean(cD5_list, axis=0));
#             cD4_mean = cD4_mean + list(np.mean(cD4_list, axis=0));
#             cD3_mean = cD3_mean + list(np.mean(cD3_list, axis=0));
#             cD2_mean = cD2_mean + list(np.mean(cD2_list, axis=0));
#             cD1_mean = cD1_mean + list(np.mean(cD1_list, axis=0));
#
#             cA5_max = cA5_max + list(np.max(cA5_list, axis=0));
#             cD5_max = cD5_max + list(np.max(cD5_list, axis=0));
#             cD4_max = cD4_max + list(np.max(cD4_list, axis=0));
#             cD3_max = cD3_max + list(np.max(cD3_list, axis=0));
#             cD2_max = cD2_max + list(np.max(cD2_list, axis=0));
#             cD1_max = cD1_max + list(np.max(cD1_list, axis=0));
#
#             cA5_min = cA5_min + list(np.min(cA5_list, axis=0));
#             cD5_min = cD5_min + list(np.min(cD5_list, axis=0));
#             cD4_min = cD4_min + list(np.min(cD4_list, axis=0));
#             cD3_min = cD3_min + list(np.min(cD3_list, axis=0));
#             cD2_min = cD2_min + list(np.min(cD2_list, axis=0));
#             cD1_min = cD1_min + list(np.min(cD1_list, axis=0));
#
#             cA5_std = cA5_std + list(np.std(cA5_list, axis=0));
#             cD5_std = cD5_std + list(np.std(cD5_list, axis=0));
#             cD4_std = cD4_std + list(np.std(cD4_list, axis=0));
#             cD3_std = cD3_std + list(np.std(cD3_list, axis=0));
#             cD2_std = cD2_std + list(np.std(cD2_list, axis=0));
#             cD1_std = cD1_std + list(np.std(cD1_list, axis=0));
#
# print(cA5_mean);print(cA5_std);print(cA5_max);print(cA5_min)
# print(len(cA5_mean))
#
# df_PNES = pd.DataFrame(list(zip(cA5_mean, cA5_std,cA5_max, cA5_min, cD5_mean, cD5_std,cD5_max, cD5_min,
#                            cD4_mean, cD4_std,cD4_max, cD4_min,cD3_mean, cD3_std,cD3_max, cD3_min,
#                            cD2_mean, cD2_std,cD2_max, cD2_min,cD1_mean, cD1_std,cD1_max, cD1_min)),
#                columns =['cA5_mean', 'cA5_std','cA5_max', 'cA5_min', 'cD5_mean', 'cD5_std','cD5_max', 'cD5_min',
#                            'cD4_mean', 'cD4_std','cD4_max', 'cD4_min','cD3_mean', 'cD3_std','cD3_max', 'cD3_min',
#                            'cD2_mean', 'cD2_std','cD2_max', 'cD2_min','cD1_mean', 'cD1_std','cD1_max', 'cD1_min'])
#
# tag_list=np.zeros(len(cA5_mean))
# df_PNES['type']=tag_list
#
# data_sum=df_ES.append(df_PNES, ignore_index=True)
#
# data_sum.to_csv('C:/Users/wxiong/Documents/PHD/2022_ML/ES_PNES_coeff_2.csv')


channels =['cA5_mean', 'cA5_std','cA5_max', 'cA5_min', 'cD5_mean', 'cD5_std','cD5_max', 'cD5_min',
          'cD4_mean', 'cD4_std','cD4_max', 'cD4_min','cD3_mean', 'cD3_std','cD3_max', 'cD3_min',
          'cD2_mean', 'cD2_std','cD2_max', 'cD2_min','cD1_mean', 'cD1_std','cD1_max', 'cD1_min',
          'type']

score_KNN_sum = []; recall_KNN_sum = []; F1_KNN_sum = [];roc_auc_KNN_sum=[]; precision_KNN_sum=[]; specificity_KNN_sum=[]
score_DT_sum = []; recall_DT_sum = []; F1_DT_sum = [];roc_auc_DT_sum=[];precision_DT_sum=[]; specificity_DT_sum=[]
score_RFT_sum = []; recall_RFT_sum = []; F1_RFT_sum = [];roc_auc_RFT_sum=[];precision_RFT_sum=[]; specificity_RFT_sum=[]
score_NB_sum = []; recall_NB_sum = []; F1_NB_sum = [];roc_auc_NB_sum=[];precision_NB_sum=[]; specificity_NB_sum=[]
score_SVM_sum = []; recall_SVM_sum = []; F1_SVM_sum = []; roc_auc_SVM_sum=[];precision_SVM_sum=[]; specificity_SVM_sum=[]
for m in range(1):
    score_KNN = [];score_DT = [];score_RFT = [];score_NB = [];score_SVM = [];
    F1_KNN = [];F1_DT = [];F1_RFT = [];F1_NB = [];F1_SVM = [];
    recall_KNN=[];recall_DT = [];recall_RFT = [];recall_NB = [];recall_SVM = [];
    roc_auc_KNN = [];roc_auc_DT = [];roc_auc_RFT = [];roc_auc_NB = [];roc_auc_SVM = [];
    precision_KNN = [];precision_DT = [];precision_RFT = [];precision_NB = [];precision_SVM = [];
    specificity_KNN = [];specificity_DT = [];specificity_RFT = [];specificity_NB = [];specificity_SVM = [];
    for i in range(100):
        dataset = pd.read_csv('C:/Users/wxiong/Documents/PHD/2022_ML/ES_PNES_coeff.csv',sep=',')
        column_name = dataset.columns[1:].tolist()
        print(column_name)

        df=dataset[channels]
        df=df.dropna()

        class_count_0, class_count_1 = df['type'].value_counts()
        print(class_count_0); print(class_count_1)
        class_0 = df[df['type'] == 1]
        class_1 = df[df['type'] == 0]
        print('class 0:', class_0.shape);
        print('class 1:', class_1.shape);
        class_0_under = class_0.sample(class_count_1)
        test_under = pd.concat([class_0_under, class_1], axis=0)

        X = test_under[channels[0:-1]]
        y = test_under[channels[-1]]

        X_train, X_validation, Y_train, Y_validation = train_test_split(X, y.values.ravel(), test_size=0.20, random_state=1)
        X_train_1 = tf.keras.utils.normalize(X_train, axis=1)
        X_validation_1 = tf.keras.utils.normalize(X_validation, axis=1)

        # Make predictions on validation dataset
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_1, Y_train)
        predictions = model.predict(X_validation_1)
        score_KNN.append(accuracy_score(Y_validation, predictions))
        recall_KNN.append(recall_score(Y_validation, predictions, average='weighted'))
        F1_KNN.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_KNN.append(roc_auc_score(Y_validation, predictions))
        precision_KNN.append(precision_score(Y_validation, predictions, average='weighted'))
        specificity_KNN.append(recall_score(np.logical_not(Y_validation), np.logical_not(predictions)))

        model = DecisionTreeClassifier()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_DT.append(accuracy_score(Y_validation, predictions))
        recall_DT.append(recall_score(Y_validation, predictions, average='weighted'))
        F1_DT.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_DT.append(roc_auc_score(Y_validation, predictions))
        precision_DT.append(precision_score(Y_validation, predictions, average='weighted'))
        specificity_DT.append(recall_score(np.logical_not(Y_validation), np.logical_not(predictions)))

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_RFT.append(accuracy_score(Y_validation, predictions))
        recall_RFT.append(recall_score(Y_validation, predictions, average='weighted'))
        F1_RFT.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_RFT.append(roc_auc_score(Y_validation, predictions))
        precision_RFT.append(precision_score(Y_validation, predictions, average='weighted'))
        specificity_RFT.append(recall_score(np.logical_not(Y_validation), np.logical_not(predictions)))

        model = GaussianNB()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_NB.append(accuracy_score(Y_validation, predictions))
        recall_NB.append(recall_score(Y_validation, predictions, average='weighted'))
        F1_NB.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_NB.append(roc_auc_score(Y_validation, predictions))
        precision_NB.append(precision_score(Y_validation, predictions, average='weighted'))
        specificity_NB.append(recall_score(np.logical_not(Y_validation), np.logical_not(predictions)))

        model = SVC(kernel='rbf')
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_SVM.append(accuracy_score(Y_validation, predictions))
        recall_SVM.append(recall_score(Y_validation, predictions, average='weighted'))
        F1_SVM.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_SVM.append(roc_auc_score(Y_validation, predictions))
        precision_SVM.append(precision_score(Y_validation, predictions, average='weighted'))
        specificity_SVM.append(recall_score(np.logical_not(Y_validation), np.logical_not(predictions)))

    score_KNN_sum.append(score_KNN);score_DT_sum.append(score_DT);score_RFT_sum.append(score_RFT);
    score_NB_sum.append(score_NB);score_SVM_sum.append(score_SVM);
    roc_auc_KNN_sum.append(roc_auc_KNN);roc_auc_DT_sum.append(roc_auc_DT);roc_auc_RFT_sum.append(roc_auc_RFT);
    roc_auc_NB_sum.append(roc_auc_NB);roc_auc_SVM_sum.append(roc_auc_SVM);
    F1_KNN_sum.append(F1_KNN);F1_DT_sum.append(F1_DT);F1_RFT_sum.append(F1_RFT);
    F1_NB_sum.append(F1_NB);F1_SVM_sum.append(F1_SVM);
    recall_KNN_sum.append(recall_KNN);recall_DT_sum.append(recall_DT);recall_RFT_sum.append(recall_RFT);
    recall_NB_sum.append(recall_NB);recall_SVM_sum.append(recall_SVM);
    precision_KNN_sum.append(precision_KNN);precision_DT_sum.append(precision_DT);precision_RFT_sum.append(precision_RFT);
    precision_NB_sum.append(precision_NB);precision_SVM_sum.append(precision_SVM);
    specificity_KNN_sum.append(specificity_KNN);specificity_DT_sum.append(specificity_DT);specificity_RFT_sum.append(specificity_RFT);
    specificity_NB_sum.append(specificity_NB);specificity_SVM_sum.append(specificity_SVM);

# print(np.mean(score_KNN_sum));print(np.mean(score_DT_sum));print(np.mean(score_RFT_sum));
# print(np.mean(score_NB_sum));print(np.mean(score_SVM_sum));

print(np.mean(score_KNN_sum));print(np.std(score_KNN_sum));
print(np.mean(recall_KNN_sum)); print(np.std(recall_KNN_sum));
print(np.mean(specificity_KNN_sum)); print(np.std(specificity_KNN_sum));

print(np.mean(score_DT_sum));print(np.std(score_DT_sum));
print(np.mean(recall_DT_sum)); print(np.std(recall_DT_sum));
print(np.mean(specificity_DT_sum)); print(np.std(specificity_DT_sum));

print(np.mean(score_RFT_sum));print(np.std(score_RFT_sum));
print(np.mean(recall_RFT_sum)); print(np.std(recall_RFT_sum));
print(np.mean(specificity_RFT_sum)); print(np.std(specificity_RFT_sum));

print(np.mean(score_NB_sum));print(np.std(score_NB_sum));
print(np.mean(recall_NB_sum)); print(np.std(recall_NB_sum));
print(np.mean(specificity_NB_sum)); print(np.std(specificity_NB_sum));

print(np.mean(score_SVM_sum));print(np.std(score_SVM_sum));
print(np.mean(recall_SVM_sum)); print(np.std(recall_SVM_sum));
print(np.mean(specificity_SVM_sum)); print(np.std(specificity_SVM_sum));


# for n in range(len(score_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=score_KNN_sum[n]
#     df['DT']=score_DT_sum[n]
#     df['RFT']=score_RFT_sum[n]
#     df['NB']=score_NB_sum[n]
#     df['SVM']=score_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong\\Documents\\PHD\\2022_ML\\DWT_score.csv')
#
#
# for n in range(len(roc_auc_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=roc_auc_KNN_sum[n]
#     df['DT']=roc_auc_DT_sum[n]
#     df['RFT']=roc_auc_RFT_sum[n]
#     df['NB']=roc_auc_NB_sum[n]
#     df['SVM']=roc_auc_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\2022_ML\\DWT_AUC.csv')
#
# for n in range(len(F1_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=F1_KNN_sum[n]
#     df['DT']=F1_DT_sum[n]
#     df['RFT']=F1_RFT_sum[n]
#     df['NB']=F1_NB_sum[n]
#     df['SVM']=F1_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\2022_ML\\DWT_F1.csv')
#
# for n in range(len(recall_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=recall_KNN_sum[n]
#     df['DT']=recall_DT_sum[n]
#     df['RFT']=recall_RFT_sum[n]
#     df['NB']=recall_NB_sum[n]
#     df['SVM']=recall_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\2022_ML\\DWT_recall.csv')
#
# for n in range(len(precision_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN'] = precision_KNN_sum[n]
#     df['DT'] = precision_DT_sum[n]
#     df['RFT'] = precision_RFT_sum[n]
#     df['NB'] = precision_NB_sum[n]
#     df['SVM'] = precision_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\2022_ML\\DWT_precision.csv')
#
# for n in range(len(specificity_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN'] = specificity_KNN_sum[n]
#     df['DT'] = specificity_DT_sum[n]
#     df['RFT'] = specificity_RFT_sum[n]
#     df['NB'] = specificity_NB_sum[n]
#     df['SVM'] = specificity_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\2022_ML\\DWT_specificity.csv')
