from __future__ import division
import mne
import numpy as np
import scipy.signal
from scipy.signal import butter, lfilter
from matplotlib import pyplot
import math
from scipy.fftpack import fft, ifft
import pandas as pd
from scipy.signal import butter, lfilter,iirfilter
from scipy.signal import hilbert
from biosppy.signals import tools
from scipy import signal
import statsmodels
from statsmodels.tsa.stattools import adfuller

def movingaverage(values, window_size):
    weights = (np.ones(window_size))/window_size
    a=np.ones(1)
    return lfilter(weights,a,values)

## detrend signal detrend signal detrend signa
# t=[0,1,2,3,4,5,6]
# ch_notch=[2,4,3,1,6,0,8]
# p = np.polyfit(t, ch_notch, 1)
# print(p)
# signal_1 = np.array(ch_notch) - np.polyval(p, t)
# print(np.polyval(p, t))
# print(signal_1)
# detrend=signal.detrend(ch_notch)
# print(detrend)
# pyplot.plot(t,ch_notch,'k')
# pyplot.plot(t,np.polyval(p, t),'r')
# pyplot.plot(t,signal_1,c=[0.5,0.5,0.5])
# pyplot.plot(t,detrend,'y--')
# pyplot.show()



from statsmodels.tsa.stattools import adfuller
### rollling mean std rollling mean std rollling mean std
# df=pd.DataFrame(np.array(ch_notch))
# rolmean = df.rolling(3).mean()
# rolstd = df.rolling(3).std()
# # Plot rolling statistics:
# pyplot.plot(df, color='blue', label='Original')
# pyplot.plot(rolmean, color='red', label='Rolling Mean')
# pyplot.plot(rolstd, color='black', label='Rolling Std')
# pyplot.legend(loc='best')
# pyplot.title('Rolling Mean & Standard Deviation')
# pyplot.show()



# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/Cz_EEGvariance_QLD0290_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/Cz_EEGauto_QLD0290_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# var_arr=[]
# for item in Raw_variance_EEG_arr:
#     if item<1e-8:
#         var_arr.append(item)
#     else:
#         var_arr.append(var_arr[-1])
# Raw_variance_EEG=var_arr
# Raw_variance_EEG=Raw_variance_EEG[0:19450]
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG,240*6)
# long_rhythm_var_arr=medium_rhythm_var_arr_3[240*6:]
# diff_EEGvar = np.diff(long_rhythm_var_arr)
#
#
# value_arr=[]
# for item in Raw_auto_EEG_arr:
#     if item<500:
#         value_arr.append(item)
#     else:
#         value_arr.append(value_arr[-1])
# Raw_auto_EEG_arr=value_arr
# Raw_auto_EEG=Raw_auto_EEG_arr[0:19450]
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG,240*6)
# long_rhythm_value_arr=medium_rhythm_value_arr_3[240*6:]
# diff_EEGauto = np.diff(long_rhythm_value_arr)
#
# t=np.linspace(0+0.00416667,0+0.00416667+0.00416667*(len(Raw_variance_EEG)-1),len(Raw_variance_EEG))
# t_window_arr=t


csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/RRI_ch21_rawvariance_QLD0290_15s_3h.csv',sep=',',header=None)
RRI_var= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/RRI_ch21_rawauto_QLD0290_15s_3h.csv',sep=',',header=None)
Raw_auto_RRI31= csv_reader.values
Raw_variance_RRI31_arr=[]
for item in RRI_var:
    Raw_variance_RRI31_arr.append(float(item))
Raw_auto_RRI31_arr=[]
for item in Raw_auto_RRI31:
    Raw_auto_RRI31_arr.append(float(item))

t=np.linspace(0+0.00416667,0+0.00416667+0.00416667*(len(Raw_variance_RRI31_arr)-1),len(Raw_variance_RRI31_arr))
t_window_arr=t

Raw_variance_RRI31=Raw_variance_RRI31_arr[0:19450]
medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31,240*6)
long_rhythm_RRI_var_arr=medium_rhythm_var_arr_3[240*6:]
Raw_auto_RRI31=Raw_auto_RRI31_arr[0:19450]
medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31,240*6)
long_rhythm_RRI_auto_arr=medium_rhythm_value_arr_3[240*6:]
diff_RRIvar = np.diff(long_rhythm_RRI_var_arr)
diff_RRIauto = np.diff(long_rhythm_RRI_auto_arr)




window_time_arr=t_window_arr[240*6:19450]

# target_arr = []
# index=[]
# for i in range(450):
#     target_arr.append(long_rhythm_var_arr[i * 40])
#     index.append(i * 40)
# diff_target_arr=np.diff(target_arr)
# from matplotlib import gridspec
# fig = pyplot.figure(figsize=(12, 10))
# gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# ax1=pyplot.subplot(gs[0])
# ax1.plot(window_time_arr,long_rhythm_var_arr,'darkblue',alpha=0.8,label='EEG variance')
# ax1.scatter(window_time_arr[index],target_arr,s=5,c='k')
# pyplot.ylabel('$\mathregular{V^2}$',fontsize=20)
# pyplot.legend(fontsize=20)
# locs, labels = pyplot.xticks(fontsize=20)
# locs, labels = pyplot.yticks(fontsize=20)
# ax2=pyplot.subplot(gs[1])
# index=index[1:]
# ax2.plot(window_time_arr[index],diff_target_arr,'darkblue',alpha=0.5, label='1st diff')
# locs, labels = pyplot.xticks(fontsize=20)
# locs, labels = pyplot.yticks(fontsize=20)
# pyplot.xlabel('Time (hours)',fontsize=20)
# pyplot.legend(fontsize=20)
# pyplot.tight_layout()
# pyplot.show()
# ch_notch=target_arr
# df=pd.DataFrame(np.array(ch_notch))
# # Perform Dickey-Fuller test:
# dftest = adfuller(df, autolag='AIC')
# print(dftest)
# dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
# for key, value in dftest[4].items():
#     dfoutput['Critical Value (%s)' % key] = value
# print(dfoutput)
# ch_notch=diff_target_arr
# df=pd.DataFrame(np.array(ch_notch))
# dftest = adfuller(df, autolag='AIC')
# print(dftest)
# dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
# for key, value in dftest[4].items():
#     dfoutput['Critical Value (%s)' % key] = value
# print(dfoutput)
#
# ch_notch=target_arr
# df=pd.DataFrame(ch_notch)
# from pandas.plotting import autocorrelation_plot
# autocorrelation_plot(df)
# pyplot.show()
# from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# plot_acf(df)
# pyplot.show()
# ### partial coorelation
# plot_pacf(df)
# pyplot.show()
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# fig, ax = plt.subplots(2,1)
# fig = sm.graphics.tsa.plot_acf(df,ax=ax[0])
# fig = sm.graphics.tsa.plot_pacf(df,ax=ax[1])
# plt.show()
#
#
#
# target_arr = []
# index=[]
# for i in range(450):
#     target_arr.append(long_rhythm_value_arr[i * 40])
#     index.append(i * 40)
# diff_target_arr=np.diff(target_arr)
#
# from matplotlib import gridspec
# fig = pyplot.figure(figsize=(12, 10))
# gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# ax1=pyplot.subplot(gs[0])
# ax1.plot(window_time_arr[index],target_arr,'darkblue',alpha=0.8,label='EEG auto')
# ax1.scatter(window_time_arr[index],target_arr,s=5,c='k')
# pyplot.legend(fontsize=20)
# locs, labels = pyplot.xticks(fontsize=20)
# locs, labels = pyplot.yticks(fontsize=20)
# ax2=pyplot.subplot(gs[1])
# index=index[1:]
# ax2.plot(window_time_arr[index],diff_target_arr,'darkblue',alpha=0.5, label='1st diff')
# locs, labels = pyplot.xticks(fontsize=20)
# locs, labels = pyplot.yticks(fontsize=20)
# pyplot.xlabel('Time (hours)',fontsize=20)
# pyplot.legend(fontsize=20)
# pyplot.tight_layout()
# pyplot.show()
#
# ch_notch=target_arr
# df=pd.DataFrame(np.array(ch_notch))
# dftest = adfuller(df, autolag='AIC')
# print(dftest)
# dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
# for key, value in dftest[4].items():
#     dfoutput['Critical Value (%s)' % key] = value
# print(dfoutput)
# ch_notch=diff_target_arr
# df=pd.DataFrame(np.array(ch_notch))
# dftest = adfuller(df, autolag='AIC')
# print(dftest)
# dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
# for key, value in dftest[4].items():
#     dfoutput['Critical Value (%s)' % key] = value
# print(dfoutput)
#
# ch_notch=target_arr
# df=pd.DataFrame(ch_notch)
# from pandas.plotting import autocorrelation_plot
# autocorrelation_plot(df)
# pyplot.show()
# from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# plot_acf(df)
# pyplot.show()
# ### partial coorelation
# plot_pacf(df)
# pyplot.show()
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# fig, ax = plt.subplots(2,1)
# fig = sm.graphics.tsa.plot_acf(df,ax=ax[0])
# fig = sm.graphics.tsa.plot_pacf(df,ax=ax[1])
# plt.show()





target_arr = []
index=[]
for i in range(450):
    target_arr.append(long_rhythm_RRI_var_arr[i * 40])
    index.append(i * 40)
diff_target_arr=np.diff(target_arr)
from matplotlib import gridspec
fig = pyplot.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
ax1=pyplot.subplot(gs[0])
ax1.plot(window_time_arr[index],target_arr,'darkblue',alpha=0.8,label='RRI var')
ax1.scatter(window_time_arr[index],target_arr,s=5,c='k')
pyplot.ylabel('$\mathregular{s^2}$',fontsize=20)
pyplot.legend(fontsize=20)
locs, labels = pyplot.xticks(fontsize=20)
locs, labels = pyplot.yticks(fontsize=20)
ax2=pyplot.subplot(gs[1])
index=index[1:]
ax2.plot(window_time_arr[index],diff_target_arr,'darkblue',alpha=0.5, label='1st diff')
locs, labels = pyplot.xticks(fontsize=20)
locs, labels = pyplot.yticks(fontsize=20)
pyplot.xlabel('Time (hours)',fontsize=20)
pyplot.legend(fontsize=20)
pyplot.tight_layout()
pyplot.show()

ch_notch=target_arr
df=pd.DataFrame(np.array(ch_notch))
dftest = adfuller(df, autolag='AIC')
print(dftest)
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value
print(dfoutput)
ch_notch=diff_target_arr
df=pd.DataFrame(np.array(ch_notch))
dftest = adfuller(df, autolag='AIC')
print(dftest)
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value
print(dfoutput)

ch_notch=target_arr
df=pd.DataFrame(ch_notch)
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df)
pyplot.show()
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(df)
pyplot.show()
plot_pacf(df)
pyplot.show()
import matplotlib.pyplot as plt
import statsmodels.api as sm
fig, ax = plt.subplots(2,1)
fig = sm.graphics.tsa.plot_acf(df,ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(df,ax=ax[1])
plt.show()






target_arr = []
index=[]
for i in range(450):
    target_arr.append(long_rhythm_RRI_auto_arr[i * 40])
    index.append(i * 40)
diff_target_arr=np.diff(target_arr)
from matplotlib import gridspec
fig = pyplot.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
ax1=pyplot.subplot(gs[0])
ax1.plot(window_time_arr[index],target_arr,'darkblue',alpha=0.8,label='RRI auto')
ax1.scatter(window_time_arr[index],target_arr,s=5,c='k')
pyplot.legend(fontsize=20)
locs, labels = pyplot.xticks(fontsize=20)
locs, labels = pyplot.yticks(fontsize=20)
ax2=pyplot.subplot(gs[1])
index=index[1:]
ax2.plot(window_time_arr[index],diff_target_arr,'darkblue',alpha=0.5, label='1st diff')
locs, labels = pyplot.xticks(fontsize=20)
locs, labels = pyplot.yticks(fontsize=20)
pyplot.xlabel('Time (hours)',fontsize=20)
pyplot.legend(fontsize=20)
pyplot.tight_layout()
pyplot.show()

ch_notch=target_arr
df=pd.DataFrame(np.array(ch_notch))
dftest = adfuller(df, autolag='AIC')
print(dftest)
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value
print(dfoutput)
ch_notch=diff_target_arr
df=pd.DataFrame(np.array(ch_notch))
dftest = adfuller(df, autolag='AIC')
print(dftest)
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value
print(dfoutput)

ch_notch=target_arr
df=pd.DataFrame(ch_notch)
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df)
pyplot.show()
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(df)
pyplot.show()
plot_pacf(df)
pyplot.show()
import matplotlib.pyplot as plt
import statsmodels.api as sm
fig, ax = plt.subplots(2,1)
fig = sm.graphics.tsa.plot_acf(df,ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(df,ax=ax[1])
plt.show()











































# from pandas.plotting import lag_plot
# from pandas import DataFrame
# ## plot lag-plot lag-plot lag-plot
# ch_notch=target_arr
# df=pd.DataFrame(np.array(ch_notch))
# series = pd.DataFrame(df.values)
# print(series)
# lag_plot(series,2)
# print(lag_plot(series))
# pyplot.show()


from pandas import concat
### calcualte correlation  correlation correlation;dataframe.corr function
### focus on values.shift(),see results
# ch_notch=[2,4,3,1,6,0,8,2,4,3,1,6,0,8]
# df=pd.DataFrame(np.array(ch_notch))
# values = pd.DataFrame(df.values)
# dataframe = concat([values.shift(1), values], axis=1)
# print(dataframe)
# dataframe.columns = ['t-1', 't+1']
# result = dataframe.corr()
# print(result)


### calcualte autocorrelation function, partial autocorrelation
# ch_notch=target_arr
# df=pd.DataFrame(ch_notch)
# from pandas.plotting import autocorrelation_plot
# autocorrelation_plot(df)
# pyplot.show()
# from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# plot_acf(df)
# pyplot.show()
# ### partial coorelation
# plot_pacf(df)
# pyplot.show()




from sklearn.metrics import mean_squared_error
##### persistent model
# ch_notch=[2,4,3,1,6,0,2,4,3,1,6]
# df=pd.DataFrame(ch_notch)
# values = pd.DataFrame(df.values)
# dataframe = concat([values.shift(1), values], axis=1)
# dataframe.columns = ['t-1', 't+1']
# # # split into train and test sets
# X = dataframe.values
# train, test = X[1:len(X) - 4], X[len(X) - 4:]
# train_X, train_y = train[:, 0], train[:, 1]
# test_X, test_y = test[:, 0], test[:, 1]
# ## persistence modelpersistence modelpersistence model
# def model_persistence(x):
#     return x
# # # walk-forward validation
# predictions = list()
# for x in test_X:
#     yhat = model_persistence(x)
#     predictions.append(yhat)
# test_score = mean_squared_error(test_y, predictions)
# print('Test MSE: %.3f' % test_score)
# pyplot.plot(test_y) # # plot predictions vs expected
# pyplot.plot(predictions, color='red')
# pyplot.show()




### autoregressive model
### remind from statsmodels.tsa.ar_model import AR is old class
from statsmodels.tsa.ar_model import AR
# ch_notch=[2,4,3,1,6,0,2,4,3,1,6]
# df=pd.DataFrame(np.array(ch_notch))
# values = pd.DataFrame(df.values)
# dataframe = concat([values.shift(1), values], axis=1)
# dataframe.columns = ['t-1', 't+1']
# X = dataframe.values
# train, test = X[1:len(X) - 4], X[len(X) - 4:]
# model = AR(train)
# model_fit = model.fit()
# print('Lag: %s' % model_fit.k_ar)
# print('Coefficients: %s' % model_fit.params)
# predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
# for i in range(len(predictions)):
# 	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
# error = mean_squared_error(test, predictions)
# print('Test MSE: %.3f' % error)
# pyplot.plot(test)
# pyplot.plot(predictions, color='red')
# pyplot.show()



# from statsmodels.tsa.ar_model import AutoReg, ar_select_order
# ch_notch=[2,4,3,1,6,0,2,4,3,1,6]
# ## see two methods
# mod = AutoReg(ch_notch,2) ## 2 is lags= 1,2; if not equals to 2, results change; compare with sel.ar_lags
# res = mod.fit()
# print(res.predict(0,30))
# # print(res.summary())
# fig = res.plot_predict()
# fig.show()
## dignostics figures
# fig = res.plot_diagnostics(res.predict(0,10))
# fig.show()


# sel = ar_select_order(ch_notch, 3,'bic')
# sel.ar_lags
# res = sel.model.fit()
# print(sel.ar_lags);
# #print(res.summary())
# fig = res.plot_predict(0,10,dynamic=True)
# fig.show()
# ## dignostics figures
# fig = res.plot_diagnostics(res.predict(0,10))
# fig.show()


# ### accurancy
# pyplot.plot(ch_notch[2:],c=[0.5,0.5,0.5])
# t=np.linspace(0,10,9)
# pyplot.plot(t,res.predict(0,10),'r')
# pyplot.show()
# test=ch_notch[2:]
# predictions=res.predict(0,10)
# error = mean_squared_error(test, predictions)
# print('Test MSE: %.3f' % error)