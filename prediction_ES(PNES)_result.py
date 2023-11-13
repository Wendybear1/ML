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

### prediction EEG only show

### each electrode
fig1, ax = pyplot.subplots(figsize=(10, 4))
import os
X_channel=[]
data_mean_1=[];data_mean_2=[];data_mean_3=[];data_mean_4=[];data_mean_5=[];
directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/performance/EEG/AUC'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
i=0
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
        # print(dataset.iloc[:,[1,2,3,4,5]])
        print(entry.path)
        data=[dataset.iloc[:,1],dataset.iloc[:,2],dataset.iloc[:,3],dataset.iloc[:,4],dataset.iloc[:,5]]
        # data_mean_1.append(np.mean(dataset.iloc[:,1]));
        # data_mean_2.append(np.mean(dataset.iloc[:, 2]));
        # data_mean_3.append(np.mean(dataset.iloc[:, 3]));
        # data_mean_4.append(np.mean(dataset.iloc[:, 4]));
        # data_mean_5.append(np.mean(dataset.iloc[:, 5]));
        data_mean_1.append(dataset.iloc[:, 1].values);
        data_mean_2.append(dataset.iloc[:, 2].values);
        data_mean_3.append(dataset.iloc[:, 3].values);
        data_mean_4.append(dataset.iloc[:, 4].values);
        data_mean_5.append(dataset.iloc[:, 5].values);
        bp1 = ax.boxplot(data, positions=[1+i, 3+i, 5+i, 7+i, 9+i], widths=2,
                         patch_artist=True, showfliers=False)
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
            if box == bp1['boxes'][4]:
                box.set(color='C0', linewidth=0.8)
                box.set(facecolor='C0')
        # for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        #     pyplot.setp(bp1[element], color='grey')

        i=i+15

locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
# locs, labels = pyplot.yticks([0.6,0.7,0.8],['0.6','0.7','0.8'],fontsize=12)
locs, labels = pyplot.xticks([5,20,35,50,65,80,95,110,125,140,155,170,185,200,215,230,245],['Fz','F3','F4', 'F7', 'F8', 'C4', 'C3', 'Cz', 'Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2'],fontsize=10)
# ax.hlines(0.5,0.5,24.5,'k',linestyles='--',alpha=0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Channels',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
ax.set_ylabel('AUC',fontsize=12)
ax.set_title('Preictal and interictal ES classification (EEG)',fontsize=12)
# ax.set_title('Preictal and interictal epileptic seizures classification (EEG)',fontsize=12)
ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3],bp1["boxes"][4]], ['KNN','DT', 'RFT','NB','SVM'], loc='lower left',fontsize=7)
pyplot.tight_layout()
pyplot.show()
print(np.mean(data_mean_1));print(np.mean(data_mean_2));print(np.mean(data_mean_3));print(np.mean(data_mean_4));print(np.mean(data_mean_5));
print(np.max(data_mean_1));print(np.max(data_mean_2));print(np.max(data_mean_3));print(np.max(data_mean_4));print(np.max(data_mean_5));
print(np.min(data_mean_1));print(np.min(data_mean_2));print(np.min(data_mean_3));print(np.min(data_mean_4));print(np.min(data_mean_5));

# import scipy.signal
# import scipy.stats as stats
# out=scipy.stats.f_oneway(data_mean_1[0],data_mean_1[1],data_mean_1[2],data_mean_1[3],
#                          data_mean_1[4],data_mean_1[5],data_mean_1[6],data_mean_1[7],
#                          data_mean_1[8],data_mean_1[9],data_mean_1[10],data_mean_1[11],
#                          data_mean_1[12],data_mean_1[13],data_mean_1[14],data_mean_1[15],
#                          data_mean_1[16],)
# print(out)
# out=scipy.stats.f_oneway(data_mean_2[0],data_mean_2[1],data_mean_2[2],data_mean_2[3],
#                          data_mean_2[4],data_mean_2[5],data_mean_2[6],data_mean_2[7],
#                          data_mean_2[8],data_mean_2[9],data_mean_2[10],data_mean_2[11],
#                          data_mean_2[12],data_mean_2[13],data_mean_2[14],data_mean_2[15],
#                          data_mean_2[16],)
# print(out)
# out=scipy.stats.f_oneway(data_mean_3[0],data_mean_3[1],data_mean_3[2],data_mean_3[3],
#                          data_mean_3[4],data_mean_3[5],data_mean_3[6],data_mean_3[7],
#                          data_mean_3[8],data_mean_3[9],data_mean_3[10],data_mean_3[11],
#                          data_mean_3[12],data_mean_3[13],data_mean_3[14],data_mean_3[15],
#                          data_mean_3[16],)
# print(out)
# out=scipy.stats.f_oneway(data_mean_4[0],data_mean_4[1],data_mean_4[2],data_mean_4[3],
#                          data_mean_4[4],data_mean_4[5],data_mean_4[6],data_mean_4[7],
#                          data_mean_4[8],data_mean_4[9],data_mean_4[10],data_mean_4[11],
#                          data_mean_4[12],data_mean_4[13],data_mean_4[14],data_mean_4[15],
#                          data_mean_4[16],)
# print(out)
# out=scipy.stats.f_oneway(data_mean_5[0],data_mean_5[1],data_mean_5[2],data_mean_5[3],
#                          data_mean_5[4],data_mean_5[5],data_mean_5[6],data_mean_5[7],
#                          data_mean_5[8],data_mean_5[9],data_mean_5[10],data_mean_5[11],
#                          data_mean_5[12],data_mean_5[13],data_mean_5[14],data_mean_5[15],
#                          data_mean_5[16],)
# print(out)



import os
X_channel=[]
fig1, ax = pyplot.subplots(figsize=(12, 4))
directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/performance/EEG/AUC'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
i=0
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        print(entry.path)
        dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
        # print(dataset.iloc[:,[1,2,3,4,5]])
        data=[dataset.iloc[:,1],dataset.iloc[:,2],dataset.iloc[:,3],dataset.iloc[:,4],dataset.iloc[:,5]]
        bp1 = ax.boxplot(data, positions=[1+i, 3+i, 5+i, 7+i, 9+i], widths=2,
                         patch_artist=True, showfliers=False)
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

            if box == bp1['boxes'][4]:
                box.set(color='C0', linewidth=0.8)
                box.set(facecolor='C0')

        i=i+30

import os
X_channel=[]
data_mean_1=[];data_mean_2=[];data_mean_3=[];data_mean_4=[];data_mean_5=[];
directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/performance/EEGECG/AUC'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
i=0
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        print(entry.path)
        dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
        # print(dataset.iloc[:,[1,2,3,4,5]])
        # data_mean_1.append(np.mean(dataset.iloc[:, 1]));
        # data_mean_2.append(np.mean(dataset.iloc[:, 2]));
        # data_mean_3.append(np.mean(dataset.iloc[:, 3]));
        # data_mean_4.append(np.mean(dataset.iloc[:, 4]));
        # data_mean_5.append(np.mean(dataset.iloc[:, 5]));
        data_mean_1.append(dataset.iloc[:, 1].values);
        data_mean_2.append(dataset.iloc[:, 2].values);
        data_mean_3.append(dataset.iloc[:, 3].values);
        data_mean_4.append(dataset.iloc[:, 4].values);
        data_mean_5.append(dataset.iloc[:, 5].values);
        data=[dataset.iloc[:,1],dataset.iloc[:,2],dataset.iloc[:,3],dataset.iloc[:,4],dataset.iloc[:,5]]

        bp2 = ax.boxplot(data, positions=[14+i, 16+i, 18+i, 20+i, 22+i], widths=2,
                         patch_artist=True, showfliers=False)
        for box in bp2['boxes']:
            if box == bp2['boxes'][0]:
                box.set(color='k', linewidth=0.8)
                box.set(facecolor='w',alpha=0.5)
            if box == bp2['boxes'][1]:
                box.set(color='r', linewidth=0.8)
                box.set(facecolor='w',alpha=0.5)
            if box == bp2['boxes'][2]:
                box.set(color='g', linewidth=0.8)
                box.set(facecolor='w', alpha=0.5)

            if box == bp2['boxes'][3]:
                box.set(color='grey', linewidth=0.8)
                box.set(facecolor='w',alpha=0.5)

            if box == bp2['boxes'][4]:
                box.set(color='C0', linewidth=0.8)
                box.set(facecolor='w',alpha=0.5)

        i=i+30

locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
locs, labels = pyplot.xticks(
    [5, 18, 35, 48, 65, 78, 95, 108, 125, 138, 155, 168, 185, 198, 215, 228, 245, 258, 275, 288, 305, 318, 335, 348,
     365, 378, 395, 408, 425, 438, 455, 468, 485,498],
    ['Fz', 'Fz_ECG', 'F3', 'F3_ECG', 'F4', 'F4_ECG', 'F7', 'F7_ECG', 'F8', 'F8_ECG', 'C4', 'C4_ECG', 'C3', 'C3_ECG', 'Cz', 'Cz_ECG', 'Pz', 'Pz_ECG', 'P3',
     'P3_ECG', 'P4', 'P4_ECG', 'T3', 'T3_ECG', 'T4', 'T4_ECG', 'T5', 'T5_ECG', 'T6',
     'T6_ECG', 'O1', 'O1_ECG', 'O2','O2_ECG'], fontsize=10,rotation=40)
# ax.hlines(0.5,0.5,24.5,'k',linestyles='--',alpha=0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Channels',fontsize=12)
ax.set_ylabel('AUC',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
ax.set_title('Preictal and interictal ES classification(EEGECG)',fontsize=12)
ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3],bp1["boxes"][4]], ['KNN','DT', 'RFT','NB','SVM'], loc='lower right',fontsize=7)
pyplot.tight_layout()
pyplot.show()

print(np.mean(data_mean_1));print(np.mean(data_mean_2));print(np.mean(data_mean_3));print(np.mean(data_mean_4));print(np.mean(data_mean_5));
print(np.max(data_mean_1));print(np.max(data_mean_2));print(np.max(data_mean_3));print(np.max(data_mean_4));print(np.max(data_mean_5));
print(np.min(data_mean_1));print(np.min(data_mean_2));print(np.min(data_mean_3));print(np.min(data_mean_4));print(np.min(data_mean_5));






### 3 electrode  ### 3 electrode ### 3 electrode
fig1, ax = pyplot.subplots(figsize=(6, 4))
import os
X_channel=[]
data_mean_1=[];data_mean_2=[];data_mean_3=[];data_mean_4=[];data_mean_5=[];
directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/performance/3EEG/AUC'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
i=0
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
        # print(dataset.iloc[:,[1,2,3,4,5]])
        print(entry.path)
        data=[dataset.iloc[:,1],dataset.iloc[:,2],dataset.iloc[:,3],dataset.iloc[:,4],dataset.iloc[:,5]]
        data_mean_1.append(np.mean(dataset.iloc[:,1]));
        data_mean_2.append(np.mean(dataset.iloc[:, 2]));
        data_mean_3.append(np.mean(dataset.iloc[:, 3]));
        data_mean_4.append(np.mean(dataset.iloc[:, 4]));
        data_mean_5.append(np.mean(dataset.iloc[:, 5]));
        bp1 = ax.boxplot(data, positions=[1+i, 3+i, 5+i, 7+i, 9+i], widths=2,
                         patch_artist=True, showfliers=False)
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
            if box == bp1['boxes'][4]:
                box.set(color='C0', linewidth=0.8)
                box.set(facecolor='C0')
        i=i+15

locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
locs, labels = pyplot.xticks([5,20,35],['Cz,C3,C4','Fz,F3,F4', 'Pz,P3,P4'],fontsize=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Channels',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
ax.set_ylabel('AUC',fontsize=12)
ax.set_title('Preictal and interictal ES classification (EEG)',fontsize=12)
ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3],bp1["boxes"][4]], ['KNN','DT', 'RFT','NB','SVM'], loc='upper left',fontsize=7)
pyplot.tight_layout()
pyplot.show()



dataset_KNN=[];dataset_DT=[];dataset_RFT=[];dataset_NB=[];dataset_SVM=[];
directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/performance/3EEG/AUC'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
i=0
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
        # print(dataset.iloc[:,[1,2,3,4,5]])
        print(entry.path)
        data=[dataset.iloc[:,1],dataset.iloc[:,2],dataset.iloc[:,3],dataset.iloc[:,4],dataset.iloc[:,5]]
        dataset_KNN.append(dataset.iloc[:,1])
        dataset_DT.append(dataset.iloc[:,2])
        dataset_RFT.append(dataset.iloc[:, 3])
        dataset_NB.append(dataset.iloc[:, 4])
        dataset_SVM.append(dataset.iloc[:, 5])

# print(dataset_KNN)
import scipy.stats as stats
# for item in dataset_DT:
#     _, p = stats.normaltest(np.array(item))
#     print(p);

from scipy.stats import f_oneway
# print(dataset_KNN[0]);print(type(dataset_KNN[0].values.tolist()));
# F, p=f_oneway(dataset_KNN[0].values.tolist(), dataset_KNN[1].values.tolist(), dataset_KNN[2].values.tolist(), dataset_KNN[3].values.tolist(), dataset_KNN[4].values.tolist())
# print(p)
# print(stats.ttest_ind(dataset_KNN[1].values, dataset_KNN[0].values))
# print(stats.ttest_ind(dataset_KNN[1].values, dataset_KNN[2].values))
# print(stats.ttest_ind(dataset_DT[1].values, dataset_DT[0].values))
# print(stats.ttest_ind(dataset_DT[1].values, dataset_DT[2].values))
# print(stats.ttest_ind(dataset_RFT[1].values, dataset_RFT[0].values))
# print(stats.ttest_ind(dataset_RFT[1].values, dataset_RFT[2].values))
# print(stats.ttest_ind(dataset_NB[1].values, dataset_NB[0].values))
# print(stats.ttest_ind(dataset_NB[1].values, dataset_NB[2].values))
# print(stats.ttest_ind(dataset_SVM[1].values, dataset_SVM[0].values))
# print(stats.ttest_ind(dataset_SVM[1].values, dataset_SVM[2].values))
#
# print(stats.ttest_1samp(dataset_SVM[0].values, 0.5))
# print(stats.ttest_1samp(dataset_SVM[1].values, 0.5))
# print(stats.ttest_1samp(dataset_SVM[2].values, 0.5))
# print(stats.ttest_1samp(dataset_NB[0].values, 0.5))
# print(stats.ttest_1samp(dataset_NB[1].values, 0.5))
# print(stats.ttest_1samp(dataset_NB[2].values, 0.5))
# print(stats.ttest_1samp(dataset_RFT[0].values, 0.5))
# print(stats.ttest_1samp(dataset_RFT[1].values, 0.5))
# print(stats.ttest_1samp(dataset_RFT[2].values, 0.5))
# print(stats.ttest_1samp(dataset_DT[0].values, 0.5))
# print(stats.ttest_1samp(dataset_DT[1].values, 0.5))
# print(stats.ttest_1samp(dataset_DT[2].values, 0.5))
# print(stats.ttest_1samp(dataset_KNN[0].values, 0.5))
# print(stats.ttest_1samp(dataset_KNN[1].values, 0.5))
# print(stats.ttest_1samp(dataset_KNN[2].values, 0.5))



fig1, ax = pyplot.subplots(figsize=(6, 4))
import os
directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/performance/3EEG/AUC'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
i=0
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
        print(entry.path)
        data=[dataset.iloc[:,1],dataset.iloc[:,2],dataset.iloc[:,3],dataset.iloc[:,4],dataset.iloc[:,5]]
        bp1 = ax.boxplot(data, positions=[1+i, 3+i, 5+i, 7+i, 9+i], widths=2,
                         patch_artist=True, showfliers=False)
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
            if box == bp1['boxes'][4]:
                box.set(color='C0', linewidth=0.8)
                box.set(facecolor='C0')
        i=i+30
directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/performance/3EEGECG/AUC'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
i=12
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
        print(entry.path)
        data=[dataset.iloc[:,1],dataset.iloc[:,2],dataset.iloc[:,3],dataset.iloc[:,4],dataset.iloc[:,5]]
        bp1 = ax.boxplot(data, positions=[1+i, 3+i, 5+i, 7+i, 9+i], widths=2,
                         patch_artist=True, showfliers=False)
        for box in bp1['boxes']:
            if box == bp1['boxes'][0]:
                box.set(color='k', linewidth=0.8)
                box.set(facecolor='w')
            if box == bp1['boxes'][1]:
                box.set(color='r', linewidth=0.8)
                box.set(facecolor='w')
            if box == bp1['boxes'][2]:
                box.set(color='g', linewidth=0.8)
                box.set(facecolor='w')
            if box == bp1['boxes'][3]:
                box.set(color='grey', linewidth=0.8)
                box.set(facecolor='w')
            if box == bp1['boxes'][4]:
                box.set(color='C0', linewidth=0.8)
                box.set(facecolor='w')
        i=i+30
locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
locs, labels = pyplot.xticks([5,20,35,50,65,80],['Cz,C3,C4','Cz,C3,C4,ECG','Fz,F3,F4', 'Fz,F3,F4, ECG', 'Pz,P3,P4','Pz,P3,P4,ECG'],fontsize=8,rotation=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Channels',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
ax.set_ylabel('AUC',fontsize=12)
ax.set_title('Preictal and interictal ES classification',fontsize=12)
pyplot.tight_layout()
pyplot.show()




dataset_KNN=[];dataset_DT=[];dataset_RFT=[];dataset_NB=[];dataset_SVM=[];
directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/performance/3EEG/AUC'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
i=0
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
        print(entry.path)
        data = [dataset.iloc[:, 1], dataset.iloc[:, 2], dataset.iloc[:, 3], dataset.iloc[:, 4], dataset.iloc[:, 5]]
        dataset_KNN.append(dataset.iloc[:, 1])
        dataset_DT.append(dataset.iloc[:, 2])
        dataset_RFT.append(dataset.iloc[:, 3])
        dataset_NB.append(dataset.iloc[:, 4])
        dataset_SVM.append(dataset.iloc[:, 5])

print(np.mean(dataset_KNN[1]))
print(np.min(dataset_KNN[1]))
print(np.max(dataset_KNN[1]))

# print(np.mean(dataset_SVM[1]))
# print(np.min(dataset_SVM[1]))
# print(np.max(dataset_SVM[1]))

import scipy.stats as stats
for item in dataset_RFT:
    _, p = stats.normaltest(np.array(item))
    # print(p);

dataset_KNN_ECG=[];dataset_DT_ECG=[];dataset_RFT_ECG=[];dataset_NB_ECG=[];dataset_SVM_ECG=[];
directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/performance/3EEGECG/AUC'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
i=0
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
        print(entry.path)
        data = [dataset.iloc[:, 1], dataset.iloc[:, 2], dataset.iloc[:, 3], dataset.iloc[:, 4], dataset.iloc[:, 5]]
        dataset_KNN_ECG.append(dataset.iloc[:, 1])
        dataset_DT_ECG.append(dataset.iloc[:, 2])
        dataset_RFT_ECG.append(dataset.iloc[:, 3])
        dataset_NB_ECG.append(dataset.iloc[:, 4])
        dataset_SVM_ECG.append(dataset.iloc[:, 5])

# for item in dataset_RFT_ECG:
#     _, p = stats.normaltest(np.array(item))
#     # print(p);
#
# print(stats.ttest_ind(dataset_KNN[0].values, dataset_KNN_ECG[0].values))
# print(stats.ttest_ind(dataset_DT[0].values, dataset_DT_ECG[0].values))
# print(stats.ttest_ind(dataset_RFT[0].values, dataset_RFT_ECG[0].values))
# print(stats.ttest_ind(dataset_NB[0].values, dataset_NB_ECG[0].values))
# print(stats.ttest_ind(dataset_SVM[0].values, dataset_SVM_ECG[0].values))
#
# print(stats.ttest_ind(dataset_KNN[1].values, dataset_KNN_ECG[1].values))
# print(stats.ttest_ind(dataset_DT[1].values, dataset_DT_ECG[1].values))
# print(stats.ttest_ind(dataset_RFT[1].values, dataset_RFT_ECG[1].values))
# print(stats.ttest_ind(dataset_NB[1].values, dataset_NB_ECG[1].values))
# print(stats.ttest_ind(dataset_SVM[1].values, dataset_SVM_ECG[1].values))
#
# print(stats.ttest_ind(dataset_KNN[2].values, dataset_KNN_ECG[2].values))
# print(stats.ttest_ind(dataset_DT[2].values, dataset_DT_ECG[2].values))
# print(stats.ttest_ind(dataset_RFT[2].values, dataset_RFT_ECG[2].values))
# print(stats.ttest_ind(dataset_NB[2].values, dataset_NB_ECG[2].values))
# print(stats.ttest_ind(dataset_SVM[2].values, dataset_SVM_ECG[2].values))

print(np.mean(dataset_RFT_ECG[1]))
print(np.min(dataset_RFT_ECG[1]))
print(np.max(dataset_RFT_ECG[1]))















### electrode clusters ### electrode clusters ### electrode clusters
fig1, ax = pyplot.subplots(figsize=(6, 4))
import os
X_channel=[]
data_mean_1=[];data_mean_2=[];data_mean_3=[];data_mean_4=[];data_mean_5=[];
directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/performance/channelsEEG/AUC'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
i=0
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
        print(entry.path)
        data=[dataset.iloc[:,1],dataset.iloc[:,2],dataset.iloc[:,3],dataset.iloc[:,4],dataset.iloc[:,5]]
        data_mean_1.append(np.mean(dataset.iloc[:,1]));
        data_mean_2.append(np.mean(dataset.iloc[:, 2]));
        data_mean_3.append(np.mean(dataset.iloc[:, 3]));
        data_mean_4.append(np.mean(dataset.iloc[:, 4]));
        data_mean_5.append(np.mean(dataset.iloc[:, 5]));
        bp1 = ax.boxplot(data, positions=[1+i, 3+i, 5+i, 7+i, 9+i], widths=2,
                         patch_artist=True, showfliers=False)
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
            if box == bp1['boxes'][4]:
                box.set(color='C0', linewidth=0.8)
                box.set(facecolor='C0')
        i=i+15

locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
locs, labels = pyplot.xticks([5,20,35,50,65,80],['Cz,C3,C4','Fz,F3,F4,F7,F8','O1,O2', 'Pz,P3,P4','T3,T4,T5,T6','All'],fontsize=10,rotation=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Channels',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
ax.set_ylabel('AUC',fontsize=12)
ax.set_title('Preictal and interictal ES classification (EEG)',fontsize=12)
ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3],bp1["boxes"][4]], ['KNN','DT', 'RFT','NB','SVM'], loc='upper left',fontsize=7)
pyplot.tight_layout()
pyplot.show()



dataset_KNN=[];dataset_DT=[];dataset_RFT=[];dataset_NB=[];dataset_SVM=[];
directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/performance/channelsEEG/AUC'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
i=0
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
        # print(dataset.iloc[:,[1,2,3,4,5]])
        print(entry.path)
        data=[dataset.iloc[:,1],dataset.iloc[:,2],dataset.iloc[:,3],dataset.iloc[:,4],dataset.iloc[:,5]]
        dataset_KNN.append(dataset.iloc[:,1])
        dataset_DT.append(dataset.iloc[:,2])
        dataset_RFT.append(dataset.iloc[:, 3])
        dataset_NB.append(dataset.iloc[:, 4])
        dataset_SVM.append(dataset.iloc[:, 5])

# print(dataset_KNN)
import scipy.stats as stats
# for item in dataset_DT:
#     _, p = stats.normaltest(np.array(item))
#     print(p);

from scipy.stats import f_oneway
# print(dataset_KNN[0]);print(type(dataset_KNN[0].values.tolist()));
# F, p=f_oneway(dataset_KNN[0].values.tolist(), dataset_KNN[1].values.tolist(), dataset_KNN[2].values.tolist(), dataset_KNN[3].values.tolist(), dataset_KNN[4].values.tolist())
# print(p)

# print(stats.ttest_ind(dataset_KNN[1].values, dataset_KNN[0].values))
# print(stats.ttest_ind(dataset_KNN[1].values, dataset_KNN[2].values))
# print(stats.ttest_ind(dataset_KNN[1].values, dataset_KNN[3].values))
# print(stats.ttest_ind(dataset_KNN[1].values, dataset_KNN[4].values))
# print(stats.ttest_ind(dataset_DT[1].values, dataset_DT[0].values))
# print(stats.ttest_ind(dataset_DT[1].values, dataset_DT[2].values))
# print(stats.ttest_ind(dataset_DT[1].values, dataset_DT[3].values))
# print(stats.ttest_ind(dataset_DT[1].values, dataset_DT[4].values))
# print(stats.ttest_ind(dataset_RFT[1].values, dataset_RFT[0].values))
# print(stats.ttest_ind(dataset_RFT[1].values, dataset_RFT[2].values))
# print(stats.ttest_ind(dataset_RFT[1].values, dataset_RFT[3].values))
# print(stats.ttest_ind(dataset_RFT[1].values, dataset_RFT[4].values))
# print(stats.ttest_ind(dataset_NB[1].values, dataset_NB[0].values))
# print(stats.ttest_ind(dataset_NB[1].values, dataset_NB[2].values))
# print(stats.ttest_ind(dataset_NB[1].values, dataset_NB[3].values))
# print(stats.ttest_ind(dataset_NB[1].values, dataset_NB[4].values))
# print(stats.ttest_ind(dataset_SVM[1].values, dataset_SVM[0].values))
# print(stats.ttest_ind(dataset_SVM[1].values, dataset_SVM[2].values))
# print(stats.ttest_ind(dataset_SVM[1].values, dataset_SVM[3].values))
# print(stats.ttest_ind(dataset_SVM[1].values, dataset_SVM[4].values))

print(stats.ttest_ind(dataset_KNN[1].values, dataset_KNN[5].values))
print(stats.ttest_ind(dataset_DT[1].values, dataset_DT[5].values))
print(stats.ttest_ind(dataset_RFT[1].values, dataset_RFT[5].values))
print(stats.ttest_ind(dataset_NB[1].values, dataset_NB[5].values))
print(stats.ttest_ind(dataset_SVM[1].values, dataset_SVM[5].values))

# print(stats.ttest_1samp(dataset_SVM[0].values, 0.5))
# print(stats.ttest_1samp(dataset_SVM[1].values, 0.5))
# print(stats.ttest_1samp(dataset_SVM[2].values, 0.5))
# print(stats.ttest_1samp(dataset_NB[0].values, 0.5))
# print(stats.ttest_1samp(dataset_NB[1].values, 0.5))
# print(stats.ttest_1samp(dataset_NB[2].values, 0.5))
# print(stats.ttest_1samp(dataset_RFT[0].values, 0.5))
# print(stats.ttest_1samp(dataset_RFT[1].values, 0.5))
# print(stats.ttest_1samp(dataset_RFT[2].values, 0.5))
# print(stats.ttest_1samp(dataset_DT[0].values, 0.5))
# print(stats.ttest_1samp(dataset_DT[1].values, 0.5))
# print(stats.ttest_1samp(dataset_DT[2].values, 0.5))
# print(stats.ttest_1samp(dataset_KNN[0].values, 0.5))
# print(stats.ttest_1samp(dataset_KNN[1].values, 0.5))
# print(stats.ttest_1samp(dataset_KNN[2].values, 0.5))



fig1, ax = pyplot.subplots(figsize=(6, 4))
import os
directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/performance/channelsEEG/AUC'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
i=0
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
        print(entry.path)
        data=[dataset.iloc[:,1],dataset.iloc[:,2],dataset.iloc[:,3],dataset.iloc[:,4],dataset.iloc[:,5]]
        bp1 = ax.boxplot(data, positions=[1+i, 3+i, 5+i, 7+i, 9+i], widths=2,
                         patch_artist=True, showfliers=False)
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
            if box == bp1['boxes'][4]:
                box.set(color='C0', linewidth=0.8)
                box.set(facecolor='C0')
        i=i+30


directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/performance/channelsEEGECG/AUC'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
i=12
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
        print(entry.path)
        data=[dataset.iloc[:,1],dataset.iloc[:,2],dataset.iloc[:,3],dataset.iloc[:,4],dataset.iloc[:,5]]
        bp1 = ax.boxplot(data, positions=[1+i, 3+i, 5+i, 7+i, 9+i], widths=2,
                         patch_artist=True, showfliers=False)
        for box in bp1['boxes']:
            if box == bp1['boxes'][0]:
                box.set(color='k', linewidth=0.8)
                box.set(facecolor='w')
            if box == bp1['boxes'][1]:
                box.set(color='r', linewidth=0.8)
                box.set(facecolor='w')
            if box == bp1['boxes'][2]:
                box.set(color='g', linewidth=0.8)
                box.set(facecolor='w')
            if box == bp1['boxes'][3]:
                box.set(color='grey', linewidth=0.8)
                box.set(facecolor='w')
            if box == bp1['boxes'][4]:
                box.set(color='C0', linewidth=0.8)
                box.set(facecolor='w')
        i=i+30


locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
locs, labels = pyplot.xticks([5,15,30,40,60,70,90,100,125,135,155,165],['Cz,C3,C4','Cz,C3,C4,ECG','Fz,F3,F4,F7,F8','Fz,F3,F4,F7,F8,ECG','O1,O2','O1,O2,ECG','Pz,P3,P4','Pz,P3,P4,ECG','T3,T4,T5,T6','T3,T4,T5,T6,ECG','All','All,ECG'],fontsize=8,rotation=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Channels',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
ax.set_ylabel('AUC',fontsize=12)
ax.set_title('Preictal and interictal ES classification',fontsize=12)
pyplot.tight_layout()
pyplot.show()



import os
dataset_KNN=[];dataset_DT=[];dataset_RFT=[];dataset_NB=[];dataset_SVM=[];
directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/performance/channelsEEG/AUC'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
i=0
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
        print(entry.path)
        data = [dataset.iloc[:, 1], dataset.iloc[:, 2], dataset.iloc[:, 3], dataset.iloc[:, 4], dataset.iloc[:, 5]]
        dataset_KNN.append(dataset.iloc[:, 1])
        dataset_DT.append(dataset.iloc[:, 2])
        dataset_RFT.append(dataset.iloc[:, 3])
        dataset_NB.append(dataset.iloc[:, 4])
        dataset_SVM.append(dataset.iloc[:, 5])

# print(np.mean(dataset_RFT[1]))
# print(np.min(dataset_RFT[1]))
# print(np.max(dataset_RFT[1]))

print(np.mean(dataset_RFT[5]))
print(np.min(dataset_RFT[5]))
print(np.max(dataset_RFT[5]))


import scipy.stats as stats
for item in dataset_RFT:
    _, p = stats.normaltest(np.array(item))
    # print(p);

dataset_KNN_ECG=[];dataset_DT_ECG=[];dataset_RFT_ECG=[];dataset_NB_ECG=[];dataset_SVM_ECG=[];
directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/performance/channelsEEGECG/AUC'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
i=0
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
        print(entry.path)
        data = [dataset.iloc[:, 1], dataset.iloc[:, 2], dataset.iloc[:, 3], dataset.iloc[:, 4], dataset.iloc[:, 5]]
        dataset_KNN_ECG.append(dataset.iloc[:, 1])
        dataset_DT_ECG.append(dataset.iloc[:, 2])
        dataset_RFT_ECG.append(dataset.iloc[:, 3])
        dataset_NB_ECG.append(dataset.iloc[:, 4])
        dataset_SVM_ECG.append(dataset.iloc[:, 5])

# for item in dataset_RFT_ECG:
#     _, p = stats.normaltest(np.array(item))
#     # print(p);
#
# print(stats.ttest_ind(dataset_KNN[0].values, dataset_KNN_ECG[0].values))
# print(stats.ttest_ind(dataset_DT[0].values, dataset_DT_ECG[0].values))
# print(stats.ttest_ind(dataset_RFT[0].values, dataset_RFT_ECG[0].values))
# print(stats.ttest_ind(dataset_NB[0].values, dataset_NB_ECG[0].values))
# print(stats.ttest_ind(dataset_SVM[0].values, dataset_SVM_ECG[0].values))
#
# print(stats.ttest_ind(dataset_KNN[1].values, dataset_KNN_ECG[1].values))
# print(stats.ttest_ind(dataset_DT[1].values, dataset_DT_ECG[1].values))
# print(stats.ttest_ind(dataset_RFT[1].values, dataset_RFT_ECG[1].values))
# print(stats.ttest_ind(dataset_NB[1].values, dataset_NB_ECG[1].values))
# print(stats.ttest_ind(dataset_SVM[1].values, dataset_SVM_ECG[1].values))
#
# print(stats.ttest_ind(dataset_KNN[2].values, dataset_KNN_ECG[2].values))
# print(stats.ttest_ind(dataset_DT[2].values, dataset_DT_ECG[2].values))
# print(stats.ttest_ind(dataset_RFT[2].values, dataset_RFT_ECG[2].values))
# print(stats.ttest_ind(dataset_NB[2].values, dataset_NB_ECG[2].values))
# print(stats.ttest_ind(dataset_SVM[2].values, dataset_SVM_ECG[2].values))
#
# print(stats.ttest_ind(dataset_KNN[3].values, dataset_KNN_ECG[3].values))
# print(stats.ttest_ind(dataset_DT[3].values, dataset_DT_ECG[3].values))
# print(stats.ttest_ind(dataset_RFT[3].values, dataset_RFT_ECG[3].values))
# print(stats.ttest_ind(dataset_NB[3].values, dataset_NB_ECG[3].values))
# print(stats.ttest_ind(dataset_SVM[3].values, dataset_SVM_ECG[3].values))
#
# print(stats.ttest_ind(dataset_KNN[4].values, dataset_KNN_ECG[4].values))
# print(stats.ttest_ind(dataset_DT[4].values, dataset_DT_ECG[4].values))
# print(stats.ttest_ind(dataset_RFT[4].values, dataset_RFT_ECG[4].values))
# print(stats.ttest_ind(dataset_NB[4].values, dataset_NB_ECG[4].values))
# print(stats.ttest_ind(dataset_SVM[4].values, dataset_SVM_ECG[4].values))

print(stats.ttest_ind(dataset_KNN[5].values, dataset_KNN_ECG[5].values))
print(stats.ttest_ind(dataset_DT[5].values, dataset_DT_ECG[5].values))
print(stats.ttest_ind(dataset_RFT[5].values, dataset_RFT_ECG[5].values))
print(stats.ttest_ind(dataset_NB[5].values, dataset_NB_ECG[5].values))
print(stats.ttest_ind(dataset_SVM[5].values, dataset_SVM_ECG[5].values))

print(stats.ttest_ind(dataset_KNN_ECG[1].values, dataset_KNN_ECG[5].values))
print(stats.ttest_ind(dataset_DT_ECG[1].values, dataset_DT_ECG[5].values))
print(stats.ttest_ind(dataset_RFT_ECG[1].values, dataset_RFT_ECG[5].values))
print(stats.ttest_ind(dataset_NB_ECG[1].values, dataset_NB_ECG[5].values))
print(stats.ttest_ind(dataset_SVM_ECG[1].values, dataset_SVM_ECG[5].values))

# print(np.mean(dataset_RFT_ECG[1]))
# print(np.min(dataset_RFT_ECG[1]))
# print(np.max(dataset_RFT_ECG[1]))

print(np.mean(dataset_RFT_ECG[5]))
print(np.min(dataset_RFT_ECG[5]))
print(np.max(dataset_RFT_ECG[5]))