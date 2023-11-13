from __future__ import division
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
from sklearn.metrics import accuracy_score,recall_score,f1_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from scipy.signal import lfilter
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import os
from scipy.stats import mannwhitneyu
import pandas as pd
import numpy as np
from matplotlib import pyplot
import scipy.stats as stats



### use all elelctrodes
dataset_15min=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_15_0min_channels_EEGperformance_Accuracy_x.csv',sep=',')
dataset_30min=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_30_15min_channels_EEGperformance_Accuracy_x.csv',sep=',')
dataset_45min=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_45_30min_channels_EEGperformance_Accuracy_x.csv',sep=',')
dataset_60min=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_60_45min_channels_EEGperformance_Accuracy_x.csv',sep=',')
dataset_75min=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_75_60min_channels_EEGperformance_Accuracy_x.csv',sep=',')
dataset_90min=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_90_75min_channels_EEGperformance_Accuracy_x.csv',sep=',')
dataset_105min=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_105_90min_channels_EEGperformance_Accuracy_x.csv',sep=',')
dataset_120min=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_120_105min_channels_EEGperformance_Accuracy_x.csv',sep=',')


dataset_15min_EEGECG=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_15_0min_channels_EEGECGperformance_Accuracy_x.csv',sep=',')
dataset_30min_EEGECG=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_30_15min_channels_EEGECGperformance_Accuracy_x.csv',sep=',')
dataset_45min_EEGECG=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_45_30min_channels_EEGECGperformance_Accuracy_x.csv',sep=',')
dataset_60min_EEGECG=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_60_45min_channels_EEGECGperformance_Accuracy_x.csv',sep=',')
dataset_75min_EEGECG=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_75_60min_channels_EEGECGperformance_Accuracy_x.csv',sep=',')

dataset_15min_ECG=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_15_0min_channels_ECGperformance_Accuracy_x.csv',sep=',')
dataset_30min_ECG=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_30_15min_channels_ECGperformance_Accuracy_x.csv',sep=',')
dataset_45min_ECG=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_45_30min_channels_ECGperformance_Accuracy_x.csv',sep=',')
dataset_60min_ECG=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_60_45min_channels_ECGperformance_Accuracy_x.csv',sep=',')
dataset_75min_ECG=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_75_60min_channels_ECGperformance_Accuracy_x.csv',sep=',')


data_15min = [dataset_15min.iloc[:, 1], dataset_15min.iloc[:, 2], dataset_15min.iloc[:, 3], dataset_15min.iloc[:, 5]]
data_30min = [dataset_30min.iloc[:, 1], dataset_30min.iloc[:, 2], dataset_30min.iloc[:, 3], dataset_30min.iloc[:, 5]]
data_45min = [dataset_45min.iloc[:, 1], dataset_45min.iloc[:, 2], dataset_45min.iloc[:, 3], dataset_45min.iloc[:, 5]]
data_60min = [dataset_60min.iloc[:, 1], dataset_60min.iloc[:, 2], dataset_60min.iloc[:, 3], dataset_60min.iloc[:, 5]]
data_75min = [dataset_75min.iloc[:, 1], dataset_75min.iloc[:, 2], dataset_75min.iloc[:, 3], dataset_75min.iloc[:, 5]]
data_90min = [dataset_90min.iloc[:, 1], dataset_90min.iloc[:, 2], dataset_90min.iloc[:, 3], dataset_90min.iloc[:, 5]]
data_105min = [dataset_105min.iloc[:, 1], dataset_105min.iloc[:, 2], dataset_105min.iloc[:, 3], dataset_105min.iloc[:, 5]]
data_120min = [dataset_120min.iloc[:, 1], dataset_120min.iloc[:, 2], dataset_120min.iloc[:, 3], dataset_120min.iloc[:, 5]]

data_15min_EEGECG = [dataset_15min_EEGECG.iloc[:, 1], dataset_15min_EEGECG.iloc[:, 2], dataset_15min_EEGECG.iloc[:, 3], dataset_15min_EEGECG.iloc[:, 5]]
data_30min_EEGECG = [dataset_30min_EEGECG.iloc[:, 1], dataset_30min_EEGECG.iloc[:, 2], dataset_30min_EEGECG.iloc[:, 3], dataset_30min_EEGECG.iloc[:, 5]]
data_45min_EEGECG = [dataset_45min_EEGECG.iloc[:, 1], dataset_45min_EEGECG.iloc[:, 2], dataset_45min_EEGECG.iloc[:, 3], dataset_45min_EEGECG.iloc[:, 5]]
data_60min_EEGECG = [dataset_60min_EEGECG.iloc[:, 1], dataset_60min_EEGECG.iloc[:, 2], dataset_60min_EEGECG.iloc[:, 3], dataset_60min_EEGECG.iloc[:, 5]]
data_75min_EEGECG = [dataset_75min_EEGECG.iloc[:, 1], dataset_75min_EEGECG.iloc[:, 2], dataset_75min_EEGECG.iloc[:, 3], dataset_75min_EEGECG.iloc[:, 5]]


data_15min_ECG = [dataset_15min_ECG.iloc[:, 1], dataset_15min_ECG.iloc[:, 2], dataset_15min_ECG.iloc[:, 3], dataset_15min_ECG.iloc[:, 5]]
data_30min_ECG = [dataset_30min_ECG.iloc[:, 1], dataset_30min_ECG.iloc[:, 2], dataset_30min_ECG.iloc[:, 3], dataset_30min_ECG.iloc[:, 5]]
data_45min_ECG = [dataset_45min_ECG.iloc[:, 1], dataset_45min_ECG.iloc[:, 2], dataset_45min_ECG.iloc[:, 3], dataset_45min_ECG.iloc[:, 5]]
data_60min_ECG = [dataset_60min_ECG.iloc[:, 1], dataset_60min_ECG.iloc[:, 2], dataset_60min_ECG.iloc[:, 3], dataset_60min_ECG.iloc[:, 5]]
data_75min_ECG = [dataset_75min_ECG.iloc[:, 1], dataset_75min_ECG.iloc[:, 2], dataset_75min_ECG.iloc[:, 3], dataset_75min_ECG.iloc[:, 5]]

# plot figures
fig1, ax = pyplot.subplots(figsize=(5, 4))
bp1 = ax.boxplot(data_15min, positions=[1,2,3,4], widths=0.6,patch_artist=True, showfliers=False)
bp2 = ax.boxplot(data_30min, positions=[7,8,9,10], widths=0.6,patch_artist=True, showfliers=False)
bp3 = ax.boxplot(data_45min , positions=[13,14,15,16], widths=0.6,patch_artist=True, showfliers=False)
bp4 = ax.boxplot(data_60min, positions=[19,20,21,22], widths=0.6,patch_artist=True, showfliers=False)
for box in bp1['boxes']:
    if box == bp1['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp1['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp1['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp1['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp1['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp3['boxes']:
    if box == bp3['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp3['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp3['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp3['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp3['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp4['boxes']:
    if box == bp4['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp4['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp4['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp4['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp4['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp2['boxes']:
    if box == bp2['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp2['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp2['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp2['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp2['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')

locs, labels = pyplot.yticks([0.5,0.7,0.9],['0.5','0.7','0.9'],fontsize=12)
# locs, labels = pyplot.yticks([0.6,0.7,0.8],['0.6','0.7','0.8'],fontsize=12)
# locs, labels = pyplot.yticks([0.75,0.8,0.85],['0.75','0.8','0.85'],fontsize=12)
locs, labels = pyplot.xticks([3,9,15,21],['15-0 min','30-15 min','45-30 min','60-45 min'],fontsize=10)
# locs, labels = pyplot.xticks([3,9,15,21],['75-60 min','90-75 min','105-90 min','120-105 min'],fontsize=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Preictal periods',fontsize=12)
ax.set_ylabel('Accuracy',fontsize=12)
ax.set_title('Classification(EEG)',fontsize=12)
ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3]], ['KNN','DT','RF','SVM'], loc='lower left',fontsize=7)
pyplot.tight_layout()
pyplot.show()

fig1, ax = pyplot.subplots(figsize=(5, 4))
bp1 = ax.boxplot(data_15min_ECG, positions=[1,2,3,4], widths=0.6,patch_artist=True, showfliers=False)
bp2 = ax.boxplot(data_30min_ECG, positions=[7,8,9,10], widths=0.6,patch_artist=True, showfliers=False)
bp3 = ax.boxplot(data_45min_ECG , positions=[13,14,15,16], widths=0.6,patch_artist=True, showfliers=False)
bp4 = ax.boxplot(data_60min_ECG, positions=[19,20,21,22], widths=0.6,patch_artist=True, showfliers=False)
for box in bp1['boxes']:
    if box == bp1['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp1['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp1['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp1['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp1['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp3['boxes']:
    if box == bp3['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp3['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp3['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp3['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp3['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp4['boxes']:
    if box == bp4['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp4['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp4['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp4['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp4['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp2['boxes']:
    if box == bp2['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp2['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp2['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp2['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp2['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
# locs, labels = pyplot.yticks([0.4,0.5,0.6],['0.4','0.5','0.6'],fontsize=12)
locs, labels = pyplot.yticks([0.5,0.7,0.9],['0.5','0.7','0.9'],fontsize=12)
# locs, labels = pyplot.yticks([0.45,0.50,0.55],['0.45','0.50','0.55'],fontsize=12)
locs, labels = pyplot.xticks([3,9,15,21],['15-0 min','30-15 min','45-30 min','60-45 min'],fontsize=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Preictal periods',fontsize=12)
ax.set_ylabel('Accuracy',fontsize=12)
ax.set_title('Classification(ECG)',fontsize=12)
# ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3],bp1["boxes"][4]], ['KNN','DT','RFT','NB','SVM'], loc='lower left',fontsize=7)
pyplot.tight_layout()
pyplot.show()

fig1, ax = pyplot.subplots(figsize=(5, 4))
bp1 = ax.boxplot(data_15min_EEGECG, positions=[1,2,3,4], widths=0.6,patch_artist=True, showfliers=False)
bp2 = ax.boxplot(data_30min_EEGECG, positions=[7,8,9,10], widths=0.6,patch_artist=True, showfliers=False)
bp3 = ax.boxplot(data_45min_EEGECG, positions=[13,14,15,16], widths=0.6,patch_artist=True, showfliers=False)
bp4 = ax.boxplot(data_60min_EEGECG, positions=[19,20,21,22], widths=0.6,patch_artist=True, showfliers=False)
for box in bp1['boxes']:
    if box == bp1['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp1['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp1['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp1['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp1['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp3['boxes']:
    if box == bp3['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp3['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp3['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp3['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp3['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp4['boxes']:
    if box == bp4['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp4['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp4['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp4['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp4['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp2['boxes']:
    if box == bp2['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp2['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp2['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp2['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp2['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')

locs, labels = pyplot.yticks([0.5,0.7,0.9],['0.5','0.7','0.9'],fontsize=12)
# locs, labels = pyplot.yticks([0.75,0.8,0.85],['0.75','0.8','0.85'],fontsize=12)
locs, labels = pyplot.xticks([3,9,15,21],['15-0 min','30-15 min','45-30 min','60-45 min'],fontsize=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Preictal periods',fontsize=12)
ax.set_ylabel('Accuracy',fontsize=12)
ax.set_title('Classification(EEGECG)',fontsize=12)
# ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3],bp1["boxes"][4]], ['KNN','DT','RFT','NB','SVM'], loc='center left',fontsize=7)
pyplot.tight_layout()
pyplot.show()




# plot figures
fig1, ax = pyplot.subplots(1,3, figsize=(8, 3))

custom_ylim = (0.5, 0.9)
pyplot.setp(ax, ylim=custom_ylim)
for axs in ax.flat:
    axs.spines['left'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.set_yticks([])
    axs.set_xticks([])


bp1 = ax[0].boxplot(data_15min, positions=[1,1.5,2,2.5], widths=0.5,patch_artist=True, showfliers=False)
bp2 = ax[0].boxplot(data_30min, positions=[6,6.5,7,7.5], widths=0.5,patch_artist=True, showfliers=False)
bp3 = ax[0].boxplot(data_45min , positions=[11,11.5,12,12.5], widths=0.5,patch_artist=True, showfliers=False)
bp4 = ax[0].boxplot(data_60min, positions=[16,16.5,17,17.5], widths=0.5,patch_artist=True, showfliers=False)
for box in bp1['boxes']:
    if box == bp1['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp1['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp1['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp1['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp1['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp3['boxes']:
    if box == bp3['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp3['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp3['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp3['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp3['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp4['boxes']:
    if box == bp4['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp4['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp4['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp4['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp4['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp2['boxes']:
    if box == bp2['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp2['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp2['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp2['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp2['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
# locs, labels = pyplot.yticks([0.6,0.7,0.8],['0.6','0.7','0.8'],fontsize=12)
ax[0].set_yticks([0.5,0.7,0.9])
ax[0].set_yticklabels(['0.5','0.7','0.9'],fontsize=12)
ax[0].set_xticks([2,7,12,17])
ax[0].set_xticklabels(['15-0 min','30-15 min','45-30 min','60-45 min'],rotation = 13,fontsize=10)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].set_xlabel('Preictal periods',fontsize=13)
ax[0].set_ylabel('Accuracy',fontsize=13)
ax[0].set_title('EEG',fontsize=12)
# ax[0].legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3],bp1["boxes"][4]], ['KNN','DT','RFT','NB','SVM'], loc='lower left',fontsize=6)
# pyplot.tight_layout()
# pyplot.show()



bp1 = ax[1].boxplot(data_15min_ECG, positions=[1,1.5,2,2.5], widths=0.5,patch_artist=True, showfliers=False)
bp2 = ax[1].boxplot(data_30min_ECG, positions=[6,6.5,7,7.5], widths=0.5,patch_artist=True, showfliers=False)
bp3 = ax[1].boxplot(data_45min_ECG , positions=[11,11.5,12,12.5], widths=0.5,patch_artist=True, showfliers=False)
bp4 = ax[1].boxplot(data_60min_ECG, positions=[16,16.5,17,17.5], widths=0.5,patch_artist=True, showfliers=False)
for box in bp1['boxes']:
    if box == bp1['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp1['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp1['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp1['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp1['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp3['boxes']:
    if box == bp3['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp3['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp3['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp3['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp3['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp4['boxes']:
    if box == bp4['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp4['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp4['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp4['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp4['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp2['boxes']:
    if box == bp2['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp2['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp2['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp2['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp2['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
# axs[1].set_yticks([0.4,0.5,0.6])
# axs[1].set_yticklabels(['0.4','0.5','0.6'],fontsize=12)
# ax[1].set_yticks([0.5,0.7,0.9])
# ax[1].set_yticklabels(['0.5','0.7','0.9'],fontsize=12)
# axs[1].set_xticks([3,9,15,21])
# axs[1].set_xticklabels(['15-0 min','30-15 min','45-30 min','60-45 min'],rotation = 15,fontsize=12)

ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['bottom'].set_visible(False)
# axs[1].set_xlabel('Preictal periods',fontsize=12)
# axs[1].set_ylabel('Accuracy',fontsize=12)
ax[1].set_title('ECG',fontsize=12)
ax[1].set_xticks([])
ax[1].legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3]], ['KNN','DT','RF','SVM'], loc='center left',fontsize=6)
# pyplot.tight_layout()
# pyplot.show()


bp1 = ax[2].boxplot(data_15min_EEGECG, positions=[1,1.5,2,2.5], widths=0.5,patch_artist=True, showfliers=False)
bp2 = ax[2].boxplot(data_30min_EEGECG, positions=[6,6.5,7,7.5], widths=0.5,patch_artist=True, showfliers=False)
bp3 = ax[2].boxplot(data_45min_EEGECG, positions=[11,11.5,12,12.5], widths=0.5,patch_artist=True, showfliers=False)
bp4 = ax[2].boxplot(data_60min_EEGECG, positions=[16,16.5,17,17.5], widths=0.5,patch_artist=True, showfliers=False)
for box in bp1['boxes']:
    if box == bp1['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp1['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp1['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp1['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp1['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp3['boxes']:
    if box == bp3['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp3['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp3['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp3['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp3['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp4['boxes']:
    if box == bp4['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp4['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp4['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp4['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp4['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
for box in bp2['boxes']:
    if box == bp2['boxes'][0]:
        box.set(color='k', linewidth=0.8)
        box.set(facecolor='k')
    if box == bp2['boxes'][1]:
        box.set(color='r', linewidth=0.8)
        box.set(facecolor='r')
    if box == bp2['boxes'][2]:
        box.set(color='g', linewidth=0.8)
        box.set(facecolor='g')
    if box == bp2['boxes'][3]:
        box.set(color='grey', linewidth=0.8)
        box.set(facecolor='grey')
    # if box == bp2['boxes'][4]:
    #     box.set(color='C0', linewidth=0.8)
    #     box.set(facecolor='C0')
# locs, labels = pyplot.yticks([0.6,0.7,0.8],['0.6','0.7','0.8'],fontsize=12)
# ax[2].set_yticks([0.5,0.7,0.9])
# ax[2].set_yticklabels(['0.5','0.7','0.9'],fontsize=12)
# axs[2].set_xticks([3,9,15,21])
# axs[2].set_xticklabels(['15-0 min','30-15 min','45-30 min','60-45 min'],rotation = 15,fontsize=12)
ax[2].spines['right'].set_visible(False)
ax[2].spines['top'].set_visible(False)
ax[2].spines['bottom'].set_visible(False)
ax[2].set_xticks([])
# axs[2].set_xlabel('Preictal periods',fontsize=12)
# axs[2].set_ylabel('Accuracy',fontsize=12)
ax[2].set_title('EEGECG',fontsize=12)
# ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3],bp1["boxes"][4]], ['KNN','DT','RFT','NB','SVM'], loc='center left',fontsize=7)
pyplot.tight_layout()
pyplot.show()