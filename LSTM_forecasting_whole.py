from __future__ import division
from numpy import array
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import numpy as np
from scipy.signal import butter, lfilter,iirfilter
import pandas as pd
from matplotlib import pyplot
from math import sqrt

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
    # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
    # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def movingaverage(values, window_size):
    weights = (np.ones(window_size))/window_size
    a=np.ones(1)
    return lfilter(weights,a,values)





# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/SA0124channels/Cz_EEGvariance_SA0124_15s_3h.csv',sep=',',header=None)
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1282/Cz_EEGvariance_QLD1282_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# Raw_var_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_var_EEG_arr.append(float(item))
# var_arr=[]
# for item in Raw_var_EEG_arr:
#     if item<1e-8:
#         var_arr.append(item*10**12)
#     else:
#         var_arr.append(var_arr[-1])
# Raw_var_EEG_arr=var_arr




# long_rhythm_var_arr=movingaverage(Raw_var_EEG_arr,240*24)
# medium_rhythm_var_arr_1=movingaverage(Raw_var_EEG_arr,240)
# medium_rhythm_var_arr_2=movingaverage(Raw_var_EEG_arr,240*3)
# medium_rhythm_var_arr_3=movingaverage(Raw_var_EEG_arr,240*6)
# medium_rhythm_var_arr_4=movingaverage(Raw_var_EEG_arr,240*12)
# short_rhythm_var_arr_plot=movingaverage(Raw_var_EEG_arr,20*1)
# print(len(long_rhythm_var_arr));print(len(medium_rhythm_var_arr_1));print(len(medium_rhythm_var_arr_2));
#
# t_window_arr=np.linspace(0,0+0.00416667*(len(Raw_var_EEG_arr)-1),len(Raw_var_EEG_arr))
#
#
# fig, (ax1,ax2,ax3) = pyplot.subplots(3,1)
# ax1.plot(t_window_arr,Raw_var_EEG_arr,'grey',alpha=0.3, label='raw')
# ax2.plot(t_window_arr,short_rhythm_var_arr_plot,'grey',alpha=0.8, label='5min')
# ax2.plot(t_window_arr,medium_rhythm_var_arr_1,'orange',alpha=0.6, label='1h')
# ax2.plot(t_window_arr,medium_rhythm_var_arr_2,'k',alpha=0.6,label='3h')
# ax3.plot(t_window_arr,medium_rhythm_var_arr_3,'C0',alpha=0.6,label='6h')
# ax3.plot(t_window_arr,medium_rhythm_var_arr_4,'g',alpha=0.6,label='12h')
# ax3.plot(t_window_arr,long_rhythm_var_arr,'r',alpha=0.6,label='24h')
# ax1.set_title('EEG variance',fontsize=12)
# ax3.set_xlabel('Time (hours)',fontsize=12)
# ax3.set_ylabel('$\mathregular{V^2}$',fontsize=12)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# ax1.legend()
# ax1.set_xticks([])
# ax2.legend()
# ax2.set_xticks([])
# ax3.legend()
# pyplot.tight_layout()
# pyplot.show()
#
#
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/SA0124channels/Cz_EEGauto_SA0124_15s_3h.csv',sep=',',header=None)
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1282/Cz_EEGauto_QLD1282_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# Raw_var_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_var_EEG_arr.append(float(item))
# var_arr=[]
# for item in Raw_var_EEG_arr:
#     if item<500:
#         var_arr.append(item)
#     else:
#         var_arr.append(var_arr[-1])
# Raw_var_EEG_arr=var_arr
#
#
# long_rhythm_var_arr=movingaverage(Raw_var_EEG_arr,240*24)
# medium_rhythm_var_arr_1=movingaverage(Raw_var_EEG_arr,240)
# medium_rhythm_var_arr_2=movingaverage(Raw_var_EEG_arr,240*3)
# medium_rhythm_var_arr_3=movingaverage(Raw_var_EEG_arr,240*6)
# medium_rhythm_var_arr_4=movingaverage(Raw_var_EEG_arr,240*12)
# short_rhythm_var_arr_plot=movingaverage(Raw_var_EEG_arr,20*1)
# print(len(long_rhythm_var_arr));print(len(medium_rhythm_var_arr_1));print(len(medium_rhythm_var_arr_2));
#
# t_window_arr=np.linspace(0,0+0.00416667*(len(Raw_var_EEG_arr)-1),len(Raw_var_EEG_arr))
#
#
# fig, (ax1,ax2,ax3) = pyplot.subplots(3,1)
# ax1.plot(t_window_arr,Raw_var_EEG_arr,'grey',alpha=0.3, label='raw')
# ax2.plot(t_window_arr,short_rhythm_var_arr_plot,'grey',alpha=0.8, label='5min')
# ax2.plot(t_window_arr,medium_rhythm_var_arr_1,'orange',alpha=0.6, label='1h')
# ax2.plot(t_window_arr,medium_rhythm_var_arr_2,'k',alpha=0.6,label='3h')
# ax3.plot(t_window_arr,medium_rhythm_var_arr_3,'C0',alpha=0.6,label='6h')
# ax3.plot(t_window_arr,medium_rhythm_var_arr_4,'g',alpha=0.6,label='12h')
# ax3.plot(t_window_arr,long_rhythm_var_arr,'r',alpha=0.6,label='24h')
# ax1.set_title('EEG auto',fontsize=12)
# ax3.set_xlabel('Time (hours)',fontsize=12)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# ax1.legend()
# ax1.set_xticks([])
# ax2.legend()
# ax2.set_xticks([])
# ax3.legend()
# pyplot.tight_layout()
# pyplot.show()
#
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/SA0124channels/RRI_ch31_rawvariance_SA0124_15s_3h.csv',sep=',',header=None)
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1282/RRI_ch31_rawvariance_QLD1282_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# Raw_var_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_var_EEG_arr.append(float(item))
# long_rhythm_var_arr=movingaverage(Raw_var_EEG_arr,240*24)
# medium_rhythm_var_arr_1=movingaverage(Raw_var_EEG_arr,240)
# medium_rhythm_var_arr_2=movingaverage(Raw_var_EEG_arr,240*3)
# medium_rhythm_var_arr_3=movingaverage(Raw_var_EEG_arr,240*6)
# medium_rhythm_var_arr_4=movingaverage(Raw_var_EEG_arr,240*12)
# short_rhythm_var_arr_plot=movingaverage(Raw_var_EEG_arr,20*1)
# print(len(long_rhythm_var_arr));print(len(medium_rhythm_var_arr_1));print(len(medium_rhythm_var_arr_2));
#
# t_window_arr=np.linspace(0,0+0.00416667*(len(Raw_var_EEG_arr)-1),len(Raw_var_EEG_arr))
#
#
# fig, (ax1,ax2,ax3) = pyplot.subplots(3,1)
# ax1.plot(t_window_arr,Raw_var_EEG_arr,'grey',alpha=0.3, label='raw')
# ax2.plot(t_window_arr,short_rhythm_var_arr_plot,'grey',alpha=0.8, label='5min')
# ax2.plot(t_window_arr,medium_rhythm_var_arr_1,'orange',alpha=0.6, label='1h')
# ax2.plot(t_window_arr,medium_rhythm_var_arr_2,'k',alpha=0.6,label='3h')
# ax3.plot(t_window_arr,medium_rhythm_var_arr_3,'C0',alpha=0.6,label='6h')
# ax3.plot(t_window_arr,medium_rhythm_var_arr_4,'g',alpha=0.6,label='12h')
# ax3.plot(t_window_arr,long_rhythm_var_arr,'r',alpha=0.6,label='24h')
# ax1.set_title('RRI variance',fontsize=12)
# ax3.set_xlabel('Time (hours)',fontsize=12)
# ax3.set_ylabel('$\mathregular{s^2}$',fontsize=12)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# ax1.legend()
# ax1.set_xticks([])
# ax2.legend()
# ax2.set_xticks([])
# ax3.legend()
# pyplot.tight_layout()
# pyplot.show()
#
#
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/SA0124channels/RRI_ch31_rawauto_SA0124_15s_3h.csv',sep=',',header=None)
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1282/RRI_ch31_rawauto_QLD1282_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# Raw_var_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_var_EEG_arr.append(float(item))
# long_rhythm_var_arr=movingaverage(Raw_var_EEG_arr,240*24)
# medium_rhythm_var_arr_1=movingaverage(Raw_var_EEG_arr,240)
# medium_rhythm_var_arr_2=movingaverage(Raw_var_EEG_arr,240*3)
# medium_rhythm_var_arr_3=movingaverage(Raw_var_EEG_arr,240*6)
# medium_rhythm_var_arr_4=movingaverage(Raw_var_EEG_arr,240*12)
# short_rhythm_var_arr_plot=movingaverage(Raw_var_EEG_arr,20*1)
# print(len(long_rhythm_var_arr));print(len(medium_rhythm_var_arr_1));print(len(medium_rhythm_var_arr_2));
#
# t_window_arr=np.linspace(0,0+0.00416667*(len(Raw_var_EEG_arr)-1),len(Raw_var_EEG_arr))
# fig, (ax1,ax2,ax3) = pyplot.subplots(3,1)
# ax1.plot(t_window_arr,Raw_var_EEG_arr,'grey',alpha=0.3, label='raw')
# ax2.plot(t_window_arr,short_rhythm_var_arr_plot,'grey',alpha=0.8, label='5min')
# ax2.plot(t_window_arr,medium_rhythm_var_arr_1,'orange',alpha=0.6, label='1h')
# ax2.plot(t_window_arr,medium_rhythm_var_arr_2,'k',alpha=0.6,label='3h')
# ax3.plot(t_window_arr,medium_rhythm_var_arr_3,'C0',alpha=0.6,label='6h')
# ax3.plot(t_window_arr,medium_rhythm_var_arr_4,'g',alpha=0.6,label='12h')
# ax3.plot(t_window_arr,long_rhythm_var_arr,'r',alpha=0.6,label='24h')
# ax1.set_title('RRI auto',fontsize=12)
# ax3.set_xlabel('Time (hours)',fontsize=12)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# ax1.legend()
# ax1.set_xticks([])
# ax2.legend()
# ax2.set_xticks([])
# ax3.legend()
# pyplot.tight_layout()
# pyplot.show()




# fore_arr_EEGvars=[]
# # # for k in range(26):
# # # # for k in range(1):
# # #     var_arr=Raw_var_EEG_arr[0:(18720+240*3*k)]
# # #     long_rhythm_var_arr=movingaverage(var_arr,240*6)
# # #     long_var_plot = long_rhythm_var_arr[(240*6+240*3*k):(18720+240*3*k)]
# # #     rolmean_short_EEGvar=long_var_plot
# # #
# # #
# # #     target_arr = []
# # #     for i in range(432):
# # #         target_arr.append(rolmean_short_EEGvar[i * 40])
# # #
# # #     raw_seq = target_arr
# # #     n_steps_in, n_steps_out = 24*6, 3*6
# # #     X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# # #
# # #     n_features = 1
# # #     X = X.reshape((X.shape[0], X.shape[1], n_features))
# # #
# # #     model = Sequential()
# # #     # model.add(LSTM(50, activation='relu', input_shape=(n_steps_in, n_features)))
# # #     model.add(LSTM(50, activation="tanh", recurrent_activation="sigmoid", return_sequences=True,
# # #                input_shape=(n_steps_in, n_features)))
# # #     model.add(LSTM(50, activation="tanh", recurrent_activation="sigmoid", ))
# # #
# # #     model.add(Dense(n_steps_out))
# # #     model.compile(optimizer='adam', loss='mse')
# # #     model.fit(X, y, epochs=200, verbose=0)
# # #
# # #     x_input = array(rolmean_short_EEGvar[-24*6:])
# # #     x_input = x_input.reshape((1, n_steps_in, n_features))
# # #     yhat = model.predict(x_input, verbose=0)
# # #     fore_arr_EEGvars.append(yhat[0])
# # # np.savetxt("activation_LSTM_Cz_EEGvar_QLD1282.csv", fore_arr_EEGvars, delimiter=",", fmt='%s')


# # convert an array of values into a dataset matrix
# def create_dataset(dataset, look_back=1):
# 	dataX, dataY = [], []
# 	for i in range(len(dataset)-look_back-1):
# 		a = dataset[i:(i+look_back), 0]
# 		dataX.append(a)
# 		dataY.append(dataset[i + look_back, 0])
# 	return np.array(dataX), np.array(dataY)
#
# # fix random seed for reproducibility
# tf.random.set_seed(7)
#
#
# # # load the dataset
# t_window_arr=np.linspace(0,0+0.00416667*(len(Raw_var_EEG_arr)-1),len(Raw_var_EEG_arr))
# medium_rhythm_var_arr_1=movingaverage(Raw_var_EEG_arr,240*6)
# # data={'EEGvariance': list(medium_rhythm_var_arr_1[0:240*24*6])}
# data={'EEGvariance': list(medium_rhythm_var_arr_1[0:20])}
# df = pd.DataFrame(data)
# dataset = df.values
# dataset = dataset.astype('float32')

# # normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
#
# # # split into train and test sets
# train_size = int(len(dataset) * 0.3)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#
# # # reshape into X=t and Y=t+1
# look_back = 1
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
# # # reshape input to be [samples, time steps, features]
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#
# print(trainX);print(trainY);
#
# # # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(4, input_shape=(1, look_back)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# # # make predictions
# trainPredict = model.predict(trainX)
# print(trainPredict)
# testPredict = model.predict(testX)
# print(testPredict)
# # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
# # calculate root mean squared error
# trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
# # shift train predictions for plotting
# trainPredictPlot = np.empty_like(dataset)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# # shift test predictions for plotting
# testPredictPlot = np.empty_like(dataset)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# # plot baseline and predictions
# pyplot.plot(scaler.inverse_transform(dataset))
# pyplot.plot(trainPredictPlot)
# pyplot.plot(testPredictPlot)
# pyplot.show()
# # np.savetxt("C:/Users/wxiong/Documents/PHD/signal1_6h_5day_1day.csv", scaler.inverse_transform(dataset), delimiter=",", fmt='%s')
# # np.savetxt("C:/Users/wxiong/Documents/PHD/signal2_6h_5day_1day.csv", trainPredictPlot, delimiter=",", fmt='%s')
# # np.savetxt("C:/Users/wxiong/Documents/PHD/signal3_6h_5day_1day.csv", testPredictPlot, delimiter=",", fmt='%s')


### multiple step forcasting


# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks
    train, test = data[1:-(240*24-1)], data[-(240*24-1):-23]
    # restructure into windows of weekly data
    train = array(split(train, len(train) / 24))
    test = array(split(test, len(test) / 24))
    return train, test


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=24):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return array(X), array(y)


# train the model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define parameters
    verbose, epochs, batch_size = 0, 70, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = Sequential()
    # model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    # model.add(Dense(100, activation='relu'))
    model.add(LSTM(4, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(1, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


# evaluate a single model
def evaluate_model(train, test, n_input):
    # fit model
    model = build_model(train, n_input)
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = array(predictions)
    print(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores



# load data
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/SA0124channels/Cz_EEGvariance_SA0124_15s_3h.csv',sep=',',header=None)
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1282/Cz_EEGvariance_QLD1282_15s_3h.csv',sep=',',header=None)
Raw_variance_EEG= csv_reader.values
Raw_var_EEG_arr=[]
for item in Raw_variance_EEG:
    Raw_var_EEG_arr.append(float(item))
var_arr=[]
for item in Raw_var_EEG_arr:
    if item<1e-8:
        var_arr.append(item*10**12)
    else:
        var_arr.append(var_arr[-1])
Raw_var_EEG_arr=var_arr
medium_rhythm_var_arr_1=movingaverage(Raw_var_EEG_arr,240*6)
# data={'EEGvariance': list(medium_rhythm_var_arr_1[0:240*24*6])}
data={'EEGvariance': list(medium_rhythm_var_arr_1[0:240*24*3])}
df = pd.DataFrame(data)
dataset = df.values

# split into train and test
train, test = split_dataset(dataset)
# evaluate model and get scores
n_input = 240
score, scores = evaluate_model(train, test, n_input)
# summarize scores
summarize_scores('lstm', score, scores)
# # plot scores
# days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
# pyplot.plot(days, scores, marker='o', label='lstm')
# pyplot.show()