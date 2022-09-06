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






# # ### show results of one classifier (SVM...)
# fig1, ax = pyplot.subplots(figsize=(10, 4))
# import os
# data_15min=[]
# data_15min_sum=[]
# directory =r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\preES_prePNES\15min\Accuracy\EEGECG'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# i=0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
#         print(entry.path)
#         data_15min.append(np.array(dataset.iloc[:,5]))
#         data_15min_sum=data_15min_sum+ list(dataset.iloc[:, 5])
#
# data_30min=[]
# data_30min_sum=[]
# directory = r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\preES_prePNES\30min\Accuracy\EEGECG'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d: d.path)
# i = 0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',', skipinitialspace=True)
#         print(entry.path)
#         data_30min.append(np.array(dataset.iloc[:, 5]))
#         data_30min_sum = data_30min_sum + list((dataset.iloc[:, 5]))
#
# data_45min=[]
# data_45min_sum=[]
# directory = r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\preES_prePNES\45min\Accuracy\EEGECG'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d: d.path)
# i = 0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',', skipinitialspace=True)
#         print(entry.path)
#         data_45min.append(np.array(dataset.iloc[:, 5]))
#         data_45min_sum=data_45min_sum + list(dataset.iloc[:, 5])
#
#
# data_60min=[]
# data_60min_sum=[]
# directory = r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\preES_prePNES\60min\Accuracy\EEGECG'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d: d.path)
# i = 0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',', skipinitialspace=True)
#         print(entry.path)
#         data_60min.append(np.array(dataset.iloc[:,5]))
#         data_60min_sum=data_60min_sum + list(dataset.iloc[:, 5])




# # find the highest mean acc electrode
# data_test = data_15min
# for i in range(len(data_test)):
#     print(np.mean(data_test[i]))
#     # _, p = stats.normaltest(np.array(data_test[i]))
#     # print(p);
#
# # test whether the electrode is significant across electrodes
# data_test = data_15min
# for i in range(16):
#     print(stats.ttest_ind(np.array(data_test[11]), np.array(data_test[i])))
#
# # test whether the electrode is significant using different preictal
# print(stats.ttest_ind(np.array(data_15min[11]), np.array(data_30min[11])))
# print(stats.ttest_ind(np.array(data_15min[11]), np.array(data_45min[11])))
# print(stats.ttest_ind(np.array(data_15min[11]), np.array(data_60min[11])))
# print(stats.ttest_ind(np.array(data_30min[11]), np.array(data_45min[11])))
# print(stats.ttest_ind(np.array(data_30min[11]), np.array(data_60min[11])))
# print(stats.ttest_ind(np.array(data_45min[11]), np.array(data_60min[11])))


# m=0
# for i in range(len(data_60min)):
#     data=[data_15min[i],data_30min[i],data_45min[i],data_60min[i]]
#
#     bp1 = ax.boxplot(data, positions=[1+m, 4+m, 7+m, 10+m], widths=2,
#                                      patch_artist=True, showfliers=False)
#     for box in bp1['boxes']:
#         if box == bp1['boxes'][0]:
#             box.set(color='C0', linewidth=0.8)
#             box.set(facecolor='w')
#         if box == bp1['boxes'][1]:
#             box.set(color='darkblue', linewidth=0.8)
#             box.set(facecolor='w')
#         if box == bp1['boxes'][2]:
#             box.set(color='tomato', linewidth=0.8)
#             box.set(facecolor='w')
#         if box == bp1['boxes'][3]:
#             box.set(color='darkred', linewidth=0.8)
#             box.set(facecolor='w')
#
#     m = m + 15
#
# locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
# locs, labels = pyplot.xticks([5,20,35,50,65,80,95,110,125,140,155,170,185,200,215,230,245,260],['Fz_ECG','F3_ECG','F4_ECG', 'F7_ECG', 'F8_ECG', 'C4_ECG', 'C3_ECG', 'Cz_ECG', 'Pz_ECG', 'P3_ECG', 'P4_ECG', 'T3_ECG', 'T4_ECG', 'T5_ECG', 'T6_ECG', 'O1_ECG', 'O2_ECG'],fontsize=10, rotation=30 )
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Electrodes',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
# ax.set_title('Preictal data classification',fontsize=12)
# ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3]], ['15-0 min','30-15 min', '45-30 min','60-45 min'], loc='lower right',fontsize=7)
# pyplot.tight_layout()
# pyplot.show()





# # plot acc across electrode in 4 boxes
# fig1, ax = pyplot.subplots()
# data_sum=[data_15min_sum,data_30min_sum,data_45min_sum,data_60min_sum]
# bp1 = ax.boxplot(data_sum, positions=[1,2,3,4], widths=0.4,patch_artist=True, showfliers=False)
# for box in bp1['boxes']:
#     if box == bp1['boxes'][0]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp1['boxes'][1]:
#         box.set(color='darkblue', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp1['boxes'][2]:
#         box.set(color='tomato', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp1['boxes'][3]:
#         box.set(color='darkred', linewidth=0.8)
#         box.set(facecolor='w')
# # ax.text(1, 0.68, '*', ha='center', va='bottom',fontsize=14)
# # ax.text(2, 0.68, '*', ha='center', va='bottom',fontsize=14)
#
# locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
#
# locs, labels = pyplot.xticks([1,2,3,4],['15-0 min','30-15 min','45-30 min','60-45 min'],fontsize=10)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Preictal periods',fontsize=12)
# ax.set_ylabel('Accuracy across channels',fontsize=12)
# ax.set_title('Preictal data classification',fontsize=12)
# # ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3]], ['15-0 min','30-15 min','45-30 min','60-45 min'], loc='lower right',fontsize=7)
# pyplot.tight_layout()
# pyplot.show()

# # find the mean of all acc across electrodes
# print(np.mean(data_15min_sum));print(np.std(data_15min_sum));
# print(np.mean(data_30min_sum));print(np.std(data_30min_sum));
# print(np.mean(data_45min_sum));print(np.std(data_45min_sum));
# print(np.mean(data_60min_sum));print(np.std(data_60min_sum));
#
# # test normal distribution
# _, p_15min = stats.normaltest(np.array(data_15min_sum))
# _, p_30min = stats.normaltest(np.array(data_30min_sum))
# _, p_45min = stats.normaltest(np.array(data_45min_sum))
# _, p_60min = stats.normaltest(np.array(data_60min_sum))
# print(p_15min);print(p_30min);print(p_45min);print(p_60min);
# # data_15min_sum is not normal distribution
# _,  res_30min = mannwhitneyu(data_15min_sum, data_30min_sum)
# _,  res_45min = mannwhitneyu(data_15min_sum, data_45min_sum)
# _,  res_60min = mannwhitneyu(data_15min_sum, data_60min_sum)
# print(res_30min);print(res_45min);print(res_60min);
# _,  res_45min = mannwhitneyu(data_30min_sum, data_45min_sum)
# _,  res_60min = mannwhitneyu(data_30min_sum, data_60min_sum)
# print(res_45min);print(res_60min);
# _,  res_60min = mannwhitneyu(data_45min_sum, data_60min_sum)
# print(res_60min);
#
#
#
#
# # # data_channels=[data_15min,data_30min,data_45min,data_60min]
# # # color=['C0','darkblue','tomato','darkred']
# # # for m in range(len(data_channels)):
# # #     data_single=data_channels[m]
# # #     colour=color[m]
# # #     for k in range(len(data_single)):
# # #         _, p = stats.normaltest(np.array(data_single[k]))
# # #         if p <0.05:
# # #             print(m)
# # #             print(k)
# # #     fig1, axs = pyplot.subplots(5,4,sharex=True,sharey=True)
# # #     # fig1.suptitle('Accuracy distribution using 45min-30min')
# # #     axs[0, 0].hist(data_single[0], 10,label='Fz',color=colour)
# # #     axs[0, 0].legend(loc='upper right',fontsize=8)
# # #     axs[0, 0].spines['right'].set_visible(False)
# # #     axs[0, 0].spines['top'].set_visible(False)
# # #     axs[0, 1].hist(data_single[1], 10,label='F3',color=colour)
# # #     axs[0, 1].legend(loc='upper right',fontsize=8)
# # #     axs[0, 1].spines['right'].set_visible(False)
# # #     axs[0, 1].spines['top'].set_visible(False)
# # #     axs[0, 2].hist(data_single[2], 10,label='F4',color=colour)
# # #     axs[0, 2].legend(loc='upper right',fontsize=8)
# # #     axs[0, 2].spines['right'].set_visible(False)
# # #     axs[0, 2].spines['top'].set_visible(False)
# # #     axs[0, 3].hist(data_single[3], 10,label='F7',color=colour)
# # #     axs[0, 3].legend(loc='upper right',fontsize=8)
# # #     axs[0, 3].spines['right'].set_visible(False)
# # #     axs[0, 3].spines['top'].set_visible(False)
# # #     axs[1, 0].hist(data_single[4], 10,label='F8',color=colour)
# # #     axs[1, 0].legend(loc='upper right',fontsize=8)
# # #     axs[1, 0].spines['right'].set_visible(False)
# # #     axs[1, 0].spines['top'].set_visible(False)
# # #     axs[1, 1].hist(data_single[5], 10,label='C4',color=colour)
# # #     axs[1, 1].legend(loc='upper right',fontsize=8)
# # #     axs[1, 1].spines['right'].set_visible(False)
# # #     axs[1, 1].spines['top'].set_visible(False)
# # #     axs[1, 2].hist(data_single[6], 10,label='C3',color=colour)
# # #     axs[1, 2].legend(loc='upper right',fontsize=8)
# # #     axs[1, 2].spines['right'].set_visible(False)
# # #     axs[1, 2].spines['top'].set_visible(False)
# # #     axs[1, 3].hist(data_single[7], 10,label='Cz',color=colour)
# # #     axs[1, 3].legend(loc='upper right',fontsize=8)
# # #     axs[1, 3].spines['right'].set_visible(False)
# # #     axs[1, 3].spines['top'].set_visible(False)
# # #     axs[2, 0].hist(data_single[8], 10,label='Pz',color=colour)
# # #     axs[2, 0].legend(loc='upper right',fontsize=8)
# # #     axs[2, 0].spines['right'].set_visible(False)
# # #     axs[2, 0].spines['top'].set_visible(False)
# # #     axs[2, 1].hist(data_single[9], 10,label='P3',color=colour)
# # #     axs[2, 1].legend(loc='upper right',fontsize=8)
# # #     axs[2, 1].spines['right'].set_visible(False)
# # #     axs[2, 1].spines['top'].set_visible(False)
# # #     axs[2, 2].hist(data_single[10], 10,label='P4',color=colour)
# # #     axs[2, 2].legend(loc='upper right',fontsize=8)
# # #     axs[2, 2].spines['right'].set_visible(False)
# # #     axs[2, 2].spines['top'].set_visible(False)
# # #     axs[2, 3].hist(data_single[11], 10,label='T3',color=colour)
# # #     axs[2, 3].legend(loc='upper right',fontsize=8)
# # #     axs[2, 3].spines['right'].set_visible(False)
# # #     axs[2, 3].spines['top'].set_visible(False)
# # #     axs[3, 0].hist(data_single[12], 10,label='T4',color=colour)
# # #     axs[3, 0].legend(loc='upper right',fontsize=8)
# # #     axs[3, 0].spines['right'].set_visible(False)
# # #     axs[3, 0].spines['top'].set_visible(False)
# # #     axs[3, 1].hist(data_single[13], 10,label='T5',color=colour)
# # #     axs[3, 1].legend(loc='upper right',fontsize=8)
# # #     axs[3, 1].spines['right'].set_visible(False)
# # #     axs[3, 1].spines['top'].set_visible(False)
# # #     axs[3, 2].hist(data_single[14], 10,label='T6',color=colour)
# # #     axs[3, 2].legend(loc='upper right',fontsize=8)
# # #     axs[3, 2].spines['right'].set_visible(False)
# # #     axs[3, 2].spines['top'].set_visible(False)
# # #     axs[3, 3].hist(data_single[15], 10,label='O1',color=colour)
# # #     axs[3, 3].legend(loc='upper right',fontsize=8)
# # #     axs[3, 3].spines['right'].set_visible(False)
# # #     axs[3, 3].spines['top'].set_visible(False)
# # #     axs[4, 0].hist(data_single[16], 10,label='O2',color=colour)
# # #     axs[4, 0].legend(loc='upper right',fontsize=8)
# # #     axs[4, 0].spines['right'].set_visible(False)
# # #     axs[4 ,0].spines['top'].set_visible(False)
# # #     axs[4, 1].set_visible(False)
# # #     axs[4, 2].set_visible(False)
# # #     axs[4, 3].set_visible(False)
# # #     pyplot.show()









# ### show results add ECG
# fig1, ax = pyplot.subplots(figsize=(10, 4))
# import os
# data_15min_EEGECG=[]
# data_15min_sum_EEGECG=[]
# directory =r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\preES_prePNES\15min\Accuracy\EEGECG'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# i=0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
#         print(entry.path)
#         data_15min_EEGECG.append(np.array(dataset.iloc[:,3]))
#         data_15min_sum_EEGECG=data_15min_sum_EEGECG+ list(dataset.iloc[:, 5])
#
# data_30min_EEGECG=[]
# data_30min_sum_EEGECG=[]
# directory = r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\preES_prePNES\30min\Accuracy\EEGECG'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d: d.path)
# i = 0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',', skipinitialspace=True)
#         print(entry.path)
#         data_30min_EEGECG.append(np.array(dataset.iloc[:, 5]))
#         data_30min_sum_EEGECG = data_30min_sum_EEGECG + list((dataset.iloc[:, 5]))
#
# data_45min_EEGECG=[]
# data_45min_sum_EEGECG=[]
# directory = r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\preES_prePNES\45min\Accuracy\EEGECG'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d: d.path)
# i = 0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',', skipinitialspace=True)
#         print(entry.path)
#         data_45min_EEGECG.append(np.array(dataset.iloc[:, 5]))
#         data_45min_sum_EEGECG=data_45min_sum_EEGECG + list(dataset.iloc[:, 5])
#
#
# data_60min_EEGECG=[]
# data_60min_sum_EEGECG=[]
# directory = r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\preES_prePNES\60min\Accuracy\EEGECG'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d: d.path)
# i = 0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',', skipinitialspace=True)
#         print(entry.path)
#         data_60min_EEGECG.append(np.array(dataset.iloc[:,3]))
#         data_60min_sum_EEGECG=data_60min_sum_EEGECG + list(dataset.iloc[:, 5])
#
# m=0
# for i in range(len(data_60min_EEGECG)):
#     data=[data_15min[i],data_30min[i],data_45min[i],data_60min[i]]
#     data_EEGECG = [data_15min_EEGECG[i], data_30min_EEGECG[i], data_45min_EEGECG[i], data_60min_EEGECG[i]]
#     bp1 = ax.boxplot(data, positions=[1+m, 4+m, 7+m, 10+m], widths=2,
#                                      patch_artist=True, showfliers=False)
#     for box in bp1['boxes']:
#         if box == bp1['boxes'][0]:
#             box.set(color='C0', linewidth=0.8)
#             box.set(facecolor='C0')
#         if box == bp1['boxes'][1]:
#             box.set(color='darkblue', linewidth=0.8)
#             box.set(facecolor='darkblue')
#         if box == bp1['boxes'][2]:
#             box.set(color='tomato', linewidth=0.8)
#             box.set(facecolor='tomato')
#         if box == bp1['boxes'][3]:
#             box.set(color='darkred', linewidth=0.8)
#             box.set(facecolor='darkred')
#
#     bp2 = ax.boxplot(data_EEGECG, positions=[12+m, 15+m, 18+m, 21+m], widths=2,
#                                      patch_artist=True, showfliers=False)
#
#     for box in bp2['boxes']:
#         if box == bp2['boxes'][0]:
#             box.set(color='C0', linewidth=0.8)
#             box.set(facecolor='w')
#         if box == bp2['boxes'][1]:
#             box.set(color='darkblue', linewidth=0.8)
#             box.set(facecolor='w')
#         if box == bp2['boxes'][2]:
#             box.set(color='tomato', linewidth=0.8)
#             box.set(facecolor='w')
#         if box == bp2['boxes'][3]:
#             box.set(color='darkred', linewidth=0.8)
#             box.set(facecolor='w')
#
#     m = m + 25
#
# locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
# locs, labels = pyplot.xticks([5,30,55,80,105,130,155,180,205,230,255,280,305,330,355,380,405,430],['Fz','F3','F4', 'F7', 'F8', 'C4', 'C3', 'Cz', 'Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2'],fontsize=10)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Channels',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
# ax.set_title('Preictal data classification',fontsize=12)
# ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3]], ['15-0 min','30-15 min', '45-30 min','60-45 min'], loc='lower right',fontsize=7)
# pyplot.tight_layout()
# pyplot.show()
#
# data_test=data_15min_EEGECG
# for i in range(len(data_test)):
#     print(np.mean(data_test[i]))
#     # _, p = stats.normaltest(np.array(data_test[i]))
#     # print(p);
# data_test = data_15min_EEGECG
# for i in range(16):
#     print(stats.ttest_ind(np.array(data_test[11]), np.array(data_test[i])))
#
# print(stats.ttest_ind(np.array(data_15min_EEGECG[11]), np.array(data_30min_EEGECG[11])))
# print(stats.ttest_ind(np.array(data_15min_EEGECG[11]), np.array(data_45min_EEGECG[11])))
# print(stats.ttest_ind(np.array(data_15min_EEGECG[11]), np.array(data_60min_EEGECG[11])))
# print(stats.ttest_ind(np.array(data_30min_EEGECG[11]), np.array(data_45min_EEGECG[11])))
# print(stats.ttest_ind(np.array(data_30min_EEGECG[11]), np.array(data_60min_EEGECG[11])))
# print(stats.ttest_ind(np.array(data_45min_EEGECG[11]), np.array(data_60min_EEGECG[11])))
#
#
# fig1, ax = pyplot.subplots()
# data_sum=[data_15min_sum,data_30min_sum,data_45min_sum,data_60min_sum]
# data_sum_EEGECG=[data_15min_sum_EEGECG,data_30min_sum_EEGECG,data_45min_sum_EEGECG,data_60min_sum_EEGECG]
# bp1 = ax.boxplot(data_sum, positions=[1,3,5,7], widths=0.4,patch_artist=True, showfliers=False)
# for box in bp1['boxes']:
#     if box == bp1['boxes'][0]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
#     if box == bp1['boxes'][1]:
#         box.set(color='darkblue', linewidth=0.8)
#         box.set(facecolor='darkblue')
#     if box == bp1['boxes'][2]:
#         box.set(color='tomato', linewidth=0.8)
#         box.set(facecolor='tomato')
#     if box == bp1['boxes'][3]:
#         box.set(color='darkred', linewidth=0.8)
#         box.set(facecolor='darkred')
# bp1 = ax.boxplot(data_sum_EEGECG, positions=[2,4,6,8], widths=0.4,patch_artist=True, showfliers=False)
# for box in bp1['boxes']:
#     if box == bp1['boxes'][0]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp1['boxes'][1]:
#         box.set(color='darkblue', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp1['boxes'][2]:
#         box.set(color='tomato', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp1['boxes'][3]:
#         box.set(color='darkred', linewidth=0.8)
#         box.set(facecolor='w')
#
# locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
# locs, labels = pyplot.xticks([1.5,3.5,5.5,7.5],['15-0 min','30-15 min','45-30 min','60-45 min'],fontsize=10)
# # ax.hlines(0.5,0.5,24.5,'k',linestyles='--',alpha=0.5)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Preictal periods',fontsize=12)
# ax.set_ylabel('Accuracy across channels',fontsize=12)
# # ax.set_ylabel('AUC',fontsize=12)
# ax.set_title('Preictal data classification',fontsize=12)
# # ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3]], ['15-0 min','30-15 min','45-30 min','60-45 min'], loc='lower right',fontsize=7)
# pyplot.tight_layout()
# pyplot.show()
#
# print(np.mean(data_15min_sum_EEGECG));print(np.std(data_15min_sum_EEGECG));
# print(np.mean(data_30min_sum_EEGECG));print(np.std(data_30min_sum_EEGECG));
# print(np.mean(data_45min_sum_EEGECG));print(np.std(data_45min_sum_EEGECG));
# print(np.mean(data_60min_sum_EEGECG));print(np.std(data_60min_sum_EEGECG));
#
# _, p_15min = stats.normaltest(np.array(data_15min_sum_EEGECG))
# _, p_30min = stats.normaltest(np.array(data_30min_sum_EEGECG))
# _, p_45min = stats.normaltest(np.array(data_45min_sum_EEGECG))
# _, p_60min = stats.normaltest(np.array(data_60min_sum_EEGECG))
# print(p_15min);print(p_30min);print(p_45min);print(p_60min);
#
# _,  res_30min = mannwhitneyu(data_15min_sum_EEGECG, data_30min_sum_EEGECG)
# _,  res_45min = mannwhitneyu(data_15min_sum_EEGECG, data_45min_sum_EEGECG)
# _,  res_60min = mannwhitneyu(data_15min_sum_EEGECG, data_60min_sum_EEGECG)
# print(res_30min);print(res_45min);print(res_60min);
# _,  res_45min = mannwhitneyu(data_30min_sum_EEGECG, data_45min_sum_EEGECG)
# _,  res_60min = mannwhitneyu(data_30min_sum_EEGECG, data_60min_sum_EEGECG)
# print(res_45min);print(res_60min);
# _,  res_60min = mannwhitneyu(data_45min_sum_EEGECG, data_60min_sum_EEGECG)
# print(res_60min);
#
# print(stats.ttest_ind(np.array(data_15min[11]), np.array(data_15min_EEGECG[11])))
# print(stats.ttest_ind(np.array(data_30min[11]), np.array(data_30min_EEGECG[11])))
# print(stats.ttest_ind(np.array(data_45min[11]), np.array(data_45min_EEGECG[11])))
# print(stats.ttest_ind(np.array(data_60min[11]), np.array(data_60min_EEGECG[11])))






# ###EEG features of 5 classifers
# fig1, ax = pyplot.subplots(figsize=(8, 4))
# import os
# X_channel=[]
# data_mean_1=[];data_mean_2=[];data_mean_3=[];data_mean_4=[];data_mean_5=[];
# directory =r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\preES_prePNES\15min\Accuracy'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# i=0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
#         # print(dataset.iloc[:,[1,2,3,4,5]])
#         print(entry.path)
#         data=[dataset.iloc[:,1],dataset.iloc[:,2],dataset.iloc[:,3],dataset.iloc[:,4],dataset.iloc[:,5]]
#         data_mean_1 = data_mean_1 + list(dataset.iloc[:, 1])
#         data_mean_2 = data_mean_2 + list(dataset.iloc[:, 2])
#         data_mean_3 = data_mean_3 + list(dataset.iloc[:, 3])
#         data_mean_4 = data_mean_4 + list(dataset.iloc[:, 4])
#         data_mean_5 = data_mean_5 + list(dataset.iloc[:, 5])
#
#         bp1 = ax.boxplot(data, positions=[1+i, 3+i, 5+i, 7+i, 9+i], widths=2,
#                          patch_artist=True, showfliers=False)
#         for box in bp1['boxes']:
#             if box == bp1['boxes'][0]:
#                 box.set(color='k', linewidth=0.8)
#                 box.set(facecolor='k')
#             if box == bp1['boxes'][1]:
#                 box.set(color='r', linewidth=0.8)
#                 box.set(facecolor='r')
#             if box == bp1['boxes'][2]:
#                 box.set(color='g', linewidth=0.8)
#                 box.set(facecolor='g')
#
#             if box == bp1['boxes'][3]:
#                 box.set(color='grey', linewidth=0.8)
#                 box.set(facecolor='grey')
#
#             if box == bp1['boxes'][4]:
#                 box.set(color='C0', linewidth=0.8)
#                 box.set(facecolor='C0')
#
#         i=i+15
#
# locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
# locs, labels = pyplot.xticks([5,20,35,50,65,80,95,110,125,140,155,170,185,200,215,230,245],['Fz','F3','F4', 'F7', 'F8', 'C4', 'C3', 'Cz', 'Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2'],fontsize=10)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Channels',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
# ax.set_title('Preictal data classification (15-0 min)',fontsize=12)
# ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3],bp1["boxes"][4]], ['KNN','DT', 'RFT','NB','SVM'], loc='lower left',fontsize=7)
# pyplot.tight_layout()
# pyplot.show()
# # print(np.mean(data_mean_1));print(np.mean(data_mean_2));print(np.mean(data_mean_3));print(np.mean(data_mean_4));print(np.mean(data_mean_5));
# # print(np.max(data_mean_1));print(np.max(data_mean_2));print(np.max(data_mean_3));print(np.max(data_mean_4));print(np.max(data_mean_5));
# # print(np.min(data_mean_1));print(np.min(data_mean_2));print(np.min(data_mean_3));print(np.min(data_mean_4));print(np.min(data_mean_5));
# #
# #
# # _, p_KNN = stats.normaltest(np.array(data_mean_1))
# # _, p_DT = stats.normaltest(np.array(data_mean_2))
# # _, p_RFT = stats.normaltest(np.array(data_mean_3))
# # _, p_NB = stats.normaltest(np.array(data_mean_4))
# # _, p_SVM = stats.normaltest(np.array(data_mean_5))
# # print(p_KNN);print(p_DT);print(p_RFT);print(p_NB);print(p_SVM);
# #
# # _,  res_1 = mannwhitneyu(data_mean_5, data_mean_1)
# # _,  res_2 = mannwhitneyu(data_mean_5, data_mean_2)
# # _,  res_3 = mannwhitneyu(data_mean_5, data_mean_3)
# # _,  res_4 = mannwhitneyu(data_mean_5, data_mean_4)
# # print(res_1);print(res_2);print(res_3);print(res_4);
#
#
#
#
# fig1, ax = pyplot.subplots(figsize=(8, 4))
# import os
# X_channel=[]
# data_mean_1=[];data_mean_2=[];data_mean_3=[];data_mean_4=[];data_mean_5=[];
# directory =r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\preES_prePNES\15min\Accuracy\EEGECG'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# i=0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
#         # print(dataset.iloc[:,[1,2,3,4,5]])
#         print(entry.path)
#         data=[dataset.iloc[:,1],dataset.iloc[:,2],dataset.iloc[:,3],dataset.iloc[:,4],dataset.iloc[:,5]]
#         data_mean_1 = data_mean_1 + list(dataset.iloc[:, 1])
#         data_mean_2 = data_mean_2 + list(dataset.iloc[:, 2])
#         data_mean_3 = data_mean_3 + list(dataset.iloc[:, 3])
#         data_mean_4 = data_mean_4 + list(dataset.iloc[:, 4])
#         data_mean_5 = data_mean_5 + list(dataset.iloc[:, 5])
#
#         bp1 = ax.boxplot(data, positions=[1+i, 3+i, 5+i, 7+i, 9+i], widths=2,
#                          patch_artist=True, showfliers=False)
#         for box in bp1['boxes']:
#             if box == bp1['boxes'][0]:
#                 box.set(color='k', linewidth=0.9)
#                 box.set(facecolor='w')
#             if box == bp1['boxes'][1]:
#                 box.set(color='r', linewidth=0.9)
#                 box.set(facecolor='w')
#             if box == bp1['boxes'][2]:
#                 box.set(color='g', linewidth=0.9)
#                 box.set(facecolor='w')
#
#             if box == bp1['boxes'][3]:
#                 box.set(color='grey', linewidth=0.9)
#                 box.set(facecolor='w')
#
#             if box == bp1['boxes'][4]:
#                 box.set(color='C0', linewidth=0.9)
#                 box.set(facecolor='w')
#
#         i=i+15
#
# locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
# locs, labels = pyplot.xticks([5,20,35,50,65,80,95,110,125,140,155,170,185,200,215,230,245],['Fz_ECG','F3_ECG','F4_ECG', 'F7_ECG', 'F8_ECG', 'C4_ECG', 'C3_ECG', 'Cz_ECG', 'Pz_ECG', 'P3_ECG', 'P4_ECG', 'T3_ECG', 'T4_ECG', 'T5_ECG', 'T6_ECG', 'O1_ECG', 'O2_ECG'],fontsize=10, rotation=30)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Channels',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
# ax.set_title('Preictal data classification (15-0 min)',fontsize=12)
# ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3],bp1["boxes"][4]], ['KNN','DT', 'RFT','NB','SVM'], loc='lower left',fontsize=7)
# pyplot.tight_layout()
# pyplot.show()
#
#
#
# # ### add ECG add ECG add ECG add ECG
# fig1, ax = pyplot.subplots(figsize=(12, 4))
# import os
# directory =r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\preES_prePNES\15min\Accuracy'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# i=0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
#         print(entry.path)
#         data=[dataset.iloc[:,1],dataset.iloc[:,2],dataset.iloc[:,3],dataset.iloc[:,4],dataset.iloc[:,5]]
#         bp1 = ax.boxplot(data, positions=[1+i, 3+i, 5+i, 7+i, 9+i], widths=2,
#                          patch_artist=True, showfliers=False)
#         for box in bp1['boxes']:
#             if box == bp1['boxes'][0]:
#                 box.set(color='k', linewidth=0.8)
#                 box.set(facecolor='k')
#             if box == bp1['boxes'][1]:
#                 box.set(color='r', linewidth=0.8)
#                 box.set(facecolor='r')
#             if box == bp1['boxes'][2]:
#                 box.set(color='g', linewidth=0.8)
#                 box.set(facecolor='g')
#             if box == bp1['boxes'][3]:
#                 box.set(color='grey', linewidth=0.8)
#                 box.set(facecolor='grey')
#             if box == bp1['boxes'][4]:
#                 box.set(color='C0', linewidth=0.8)
#                 box.set(facecolor='C0')
#
#         i=i+30
#
# data_mean_1=[];data_mean_2=[];data_mean_3=[];data_mean_4=[];data_mean_5=[];
# directory =r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\preES_prePNES\15min\Accuracy\EEGECG'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# i=0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
#         print(entry.path)
#         data=[dataset.iloc[:,1],dataset.iloc[:,2],dataset.iloc[:,3],dataset.iloc[:,4],dataset.iloc[:,5]]
#         data_mean_1 = data_mean_1 + list(dataset.iloc[:, 1])
#         data_mean_2 = data_mean_2 + list(dataset.iloc[:, 2])
#         data_mean_3 = data_mean_3 + list(dataset.iloc[:, 3])
#         data_mean_4 = data_mean_4 + list(dataset.iloc[:, 4])
#         data_mean_5 = data_mean_5 + list(dataset.iloc[:, 5])
#         bp2 = ax.boxplot(data, positions=[14+i, 16+i, 18+i, 20+i, 22+i], widths=2,
#                          patch_artist=True, showfliers=False)
#         for box in bp2['boxes']:
#             if box == bp2['boxes'][0]:
#                 box.set(color='k', linewidth=0.8)
#                 box.set(facecolor='w')
#             if box == bp2['boxes'][1]:
#                 box.set(color='r', linewidth=0.8)
#                 box.set(facecolor='w')
#             if box == bp2['boxes'][2]:
#                 box.set(color='g', linewidth=0.8)
#                 box.set(facecolor='w')
#
#             if box == bp2['boxes'][3]:
#                 box.set(color='grey', linewidth=0.8)
#                 box.set(facecolor='w')
#
#             if box == bp2['boxes'][4]:
#                 box.set(color='C0', linewidth=0.8)
#                 box.set(facecolor='w')
#         i=i+30
#
# locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
# locs, labels = pyplot.xticks(
#     [5, 18, 35, 48, 65, 78, 95, 108, 125, 138, 155, 168, 185, 198, 215, 228, 245, 258, 275, 288, 305, 318, 335, 348,
#      365, 378, 395, 408, 425, 438, 455, 468, 485, 498],
#     ['Fz', 'Fz_ECG', 'F3', 'F3_ECG', 'F4', 'F4_ECG', 'F7', 'F7_ECG', 'F8', 'F8_ECG', 'C4', 'C4_ECG', 'C3', 'C3_ECG', 'Cz', 'Cz_ECG', 'Pz', 'Pz_ECG', 'P3',
#      'P3_ECG', 'P4', 'P4_ECG', 'T3', 'T3_ECG', 'T4', 'T4_ECG', 'T5', 'T5_ECG', 'T6',
#      'T6_ECG', 'O1', 'O1_ECG', 'O2','O2_ECG'], fontsize=10,rotation=30)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Channels',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
# ax.set_title('Preictal data classification (15-0 min)',fontsize=12)
# pyplot.tight_layout()
# pyplot.show()
#
# # print(np.mean(data_mean_1));print(np.mean(data_mean_2));print(np.mean(data_mean_3));print(np.mean(data_mean_4));print(np.mean(data_mean_5));
# # print(np.max(data_mean_1));print(np.max(data_mean_2));print(np.max(data_mean_3));print(np.max(data_mean_4));print(np.max(data_mean_5));
# # print(np.min(data_mean_1));print(np.min(data_mean_2));print(np.min(data_mean_3));print(np.min(data_mean_4));print(np.min(data_mean_5));
# #
# # _, p_KNN = stats.normaltest(np.array(data_mean_1))
# # _, p_DT = stats.normaltest(np.array(data_mean_2))
# # _, p_RFT = stats.normaltest(np.array(data_mean_3))
# # _, p_NB = stats.normaltest(np.array(data_mean_4))
# # _, p_SVM = stats.normaltest(np.array(data_mean_5))
# # print(p_KNN);print(p_DT);print(p_RFT);print(p_NB);print(p_SVM);
# # _,  res_1 = mannwhitneyu(data_mean_5, data_mean_1)
# # _,  res_2 = mannwhitneyu(data_mean_5, data_mean_2)
# # _,  res_3 = mannwhitneyu(data_mean_5, data_mean_3)
# # _,  res_4 = mannwhitneyu(data_mean_5, data_mean_4)
# # print(res_1);print(res_2);print(res_3);print(res_4);











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


data_15min = [dataset_15min.iloc[:, 1], dataset_15min.iloc[:, 2], dataset_15min.iloc[:, 3], dataset_15min.iloc[:, 4], dataset_15min.iloc[:, 5]]
data_30min = [dataset_30min.iloc[:, 1], dataset_30min.iloc[:, 2], dataset_30min.iloc[:, 3], dataset_30min.iloc[:, 4], dataset_30min.iloc[:, 5]]
data_45min = [dataset_45min.iloc[:, 1], dataset_45min.iloc[:, 2], dataset_45min.iloc[:, 3], dataset_45min.iloc[:, 4], dataset_45min.iloc[:, 5]]
data_60min = [dataset_60min.iloc[:, 1], dataset_60min.iloc[:, 2], dataset_60min.iloc[:, 3], dataset_60min.iloc[:, 4], dataset_60min.iloc[:, 5]]
data_75min = [dataset_75min.iloc[:, 1], dataset_75min.iloc[:, 2], dataset_75min.iloc[:, 3], dataset_75min.iloc[:, 4], dataset_75min.iloc[:, 5]]
data_90min = [dataset_90min.iloc[:, 1], dataset_90min.iloc[:, 2], dataset_90min.iloc[:, 3], dataset_90min.iloc[:, 4], dataset_90min.iloc[:, 5]]
data_105min = [dataset_105min.iloc[:, 1], dataset_105min.iloc[:, 2], dataset_105min.iloc[:, 3], dataset_105min.iloc[:, 4], dataset_105min.iloc[:, 5]]
data_120min = [dataset_120min.iloc[:, 1], dataset_120min.iloc[:, 2], dataset_120min.iloc[:, 3], dataset_120min.iloc[:, 4], dataset_120min.iloc[:, 5]]

data_15min_EEGECG = [dataset_15min_EEGECG.iloc[:, 1], dataset_15min_EEGECG.iloc[:, 2], dataset_15min_EEGECG.iloc[:, 3], dataset_15min_EEGECG.iloc[:, 4], dataset_15min_EEGECG.iloc[:, 5]]
data_30min_EEGECG = [dataset_30min_EEGECG.iloc[:, 1], dataset_30min_EEGECG.iloc[:, 2], dataset_30min_EEGECG.iloc[:, 3], dataset_30min_EEGECG.iloc[:, 4], dataset_30min_EEGECG.iloc[:, 5]]
data_45min_EEGECG = [dataset_45min_EEGECG.iloc[:, 1], dataset_45min_EEGECG.iloc[:, 2], dataset_45min_EEGECG.iloc[:, 3], dataset_45min_EEGECG.iloc[:, 4], dataset_45min_EEGECG.iloc[:, 5]]
data_60min_EEGECG = [dataset_60min_EEGECG.iloc[:, 1], dataset_60min_EEGECG.iloc[:, 2], dataset_60min_EEGECG.iloc[:, 3], dataset_60min_EEGECG.iloc[:, 4], dataset_60min_EEGECG.iloc[:, 5]]
data_75min_EEGECG = [dataset_75min_EEGECG.iloc[:, 1], dataset_75min_EEGECG.iloc[:, 2], dataset_75min_EEGECG.iloc[:, 3], dataset_75min_EEGECG.iloc[:, 4], dataset_75min_EEGECG.iloc[:, 5]]

data_15min_ECG = [dataset_15min_ECG.iloc[:, 1], dataset_15min_ECG.iloc[:, 2], dataset_15min_ECG.iloc[:, 3], dataset_15min_ECG.iloc[:, 4], dataset_15min_ECG.iloc[:, 5]]
data_30min_ECG = [dataset_30min_ECG.iloc[:, 1], dataset_30min_ECG.iloc[:, 2], dataset_30min_ECG.iloc[:, 3], dataset_30min_ECG.iloc[:, 4], dataset_30min_ECG.iloc[:, 5]]
data_45min_ECG = [dataset_45min_ECG.iloc[:, 1], dataset_45min_ECG.iloc[:, 2], dataset_45min_ECG.iloc[:, 3], dataset_45min_ECG.iloc[:, 4], dataset_45min_ECG.iloc[:, 5]]
data_60min_ECG = [dataset_60min_ECG.iloc[:, 1], dataset_60min_ECG.iloc[:, 2], dataset_60min_ECG.iloc[:, 3], dataset_60min_ECG.iloc[:, 4], dataset_60min_ECG.iloc[:, 5]]
data_75min_ECG = [dataset_75min_ECG.iloc[:, 1], dataset_75min_ECG.iloc[:, 2], dataset_75min_ECG.iloc[:, 3], dataset_75min_ECG.iloc[:, 4], dataset_75min_ECG.iloc[:, 5]]


# # plot figures
# fig1, ax = pyplot.subplots(figsize=(5, 4))
# bp1 = ax.boxplot(data_15min, positions=[1,2,3,4,5], widths=0.3,patch_artist=True, showfliers=False)
# bp2 = ax.boxplot(data_30min, positions=[7,8,9,10,11], widths=0.3,patch_artist=True, showfliers=False)
# bp3 = ax.boxplot(data_45min , positions=[13,14,15,16,17], widths=0.3,patch_artist=True, showfliers=False)
# bp4 = ax.boxplot(data_60min, positions=[19,20,21,22,23], widths=0.3,patch_artist=True, showfliers=False)
# for box in bp1['boxes']:
#     if box == bp1['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp1['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp1['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp1['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp1['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp3['boxes']:
#     if box == bp3['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp3['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp3['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp3['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp3['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp4['boxes']:
#     if box == bp4['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp4['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp4['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp4['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp4['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp2['boxes']:
#     if box == bp2['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp2['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp2['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp2['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp2['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
#
# locs, labels = pyplot.yticks([0.5,0.7,0.9],['0.5','0.7','0.9'],fontsize=12)
# # locs, labels = pyplot.yticks([0.6,0.7,0.8],['0.6','0.7','0.8'],fontsize=12)
# # locs, labels = pyplot.yticks([0.75,0.8,0.85],['0.75','0.8','0.85'],fontsize=12)
# locs, labels = pyplot.xticks([3,9,15,21],['15-0 min','30-15 min','45-30 min','60-45 min'],fontsize=10)
# # locs, labels = pyplot.xticks([3,9,15,21],['75-60 min','90-75 min','105-90 min','120-105 min'],fontsize=10)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Preictal periods',fontsize=12)
# ax.set_ylabel('F1 score',fontsize=12)
# ax.set_title('Classification(EEG)',fontsize=12)
# ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3],bp1["boxes"][4]], ['KNN','DT','RF','NB','SVM'], loc='lower left',fontsize=7)
# pyplot.tight_layout()
# pyplot.show()
#
# fig1, ax = pyplot.subplots(figsize=(5, 4))
# bp1 = ax.boxplot(data_15min_ECG, positions=[1,2,3,4,5], widths=0.3,patch_artist=True, showfliers=False)
# bp2 = ax.boxplot(data_30min_ECG, positions=[7,8,9,10,11], widths=0.3,patch_artist=True, showfliers=False)
# bp3 = ax.boxplot(data_45min_ECG , positions=[13,14,15,16,17], widths=0.3,patch_artist=True, showfliers=False)
# bp4 = ax.boxplot(data_60min_ECG, positions=[19,20,21,22,23], widths=0.3,patch_artist=True, showfliers=False)
# for box in bp1['boxes']:
#     if box == bp1['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp1['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp1['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp1['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp1['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp3['boxes']:
#     if box == bp3['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp3['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp3['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp3['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp3['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp4['boxes']:
#     if box == bp4['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp4['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp4['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp4['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp4['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp2['boxes']:
#     if box == bp2['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp2['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp2['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp2['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp2['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# # locs, labels = pyplot.yticks([0.4,0.5,0.6],['0.4','0.5','0.6'],fontsize=12)
# locs, labels = pyplot.yticks([0.4,0.5,0.6],['0.5','0.7','0.9'],fontsize=12)
# # locs, labels = pyplot.yticks([0.45,0.50,0.55],['0.45','0.50','0.55'],fontsize=12)
# locs, labels = pyplot.xticks([3,9,15,21],['15-0 min','30-15 min','45-30 min','60-45 min'],fontsize=10)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Preictal periods',fontsize=12)
# ax.set_ylabel('F1 score',fontsize=12)
# ax.set_title('Classification(ECG)',fontsize=12)
# # ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3],bp1["boxes"][4]], ['KNN','DT','RFT','NB','SVM'], loc='lower left',fontsize=7)
# pyplot.tight_layout()
# pyplot.show()
#
# fig1, ax = pyplot.subplots(figsize=(5, 4))
# bp1 = ax.boxplot(data_15min_EEGECG, positions=[1,2,3,4,5], widths=0.3,patch_artist=True, showfliers=False)
# bp2 = ax.boxplot(data_30min_EEGECG, positions=[7,8,9,10,11], widths=0.3,patch_artist=True, showfliers=False)
# bp3 = ax.boxplot(data_45min_EEGECG, positions=[13,14,15,16,17], widths=0.3,patch_artist=True, showfliers=False)
# bp4 = ax.boxplot(data_60min_EEGECG, positions=[19,20,21,22,23], widths=0.3,patch_artist=True, showfliers=False)
# for box in bp1['boxes']:
#     if box == bp1['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp1['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp1['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp1['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp1['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp3['boxes']:
#     if box == bp3['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp3['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp3['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp3['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp3['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp4['boxes']:
#     if box == bp4['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp4['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp4['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp4['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp4['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp2['boxes']:
#     if box == bp2['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp2['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp2['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp2['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp2['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
#
# locs, labels = pyplot.yticks([0.5,0.7,0.9],['0.5','0.7','0.9'],fontsize=12)
# # locs, labels = pyplot.yticks([0.75,0.8,0.85],['0.75','0.8','0.85'],fontsize=12)
# locs, labels = pyplot.xticks([3,9,15,21],['15-0 min','30-15 min','45-30 min','60-45 min'],fontsize=10)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Preictal periods',fontsize=12)
# ax.set_ylabel('F1 score',fontsize=12)
# ax.set_title('Classification(EEGECG)',fontsize=12)
# # ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3],bp1["boxes"][4]], ['KNN','DT','RFT','NB','SVM'], loc='center left',fontsize=7)
# pyplot.tight_layout()
# pyplot.show()


# for m in range(1,6):
#     print(np.mean(dataset_15min.iloc[:,m]));print(np.std(dataset_15min.iloc[:,m]));


# for m in range(1,6):
#     print(np.median(dataset_15min.iloc[:,m]));print(np.median(dataset_30min.iloc[:,m]));
#     print(np.median(dataset_45min.iloc[:,m]));print(np.median(dataset_60min.iloc[:,m]));
# normal distribution
#     _, p_15min = stats.normaltest(np.array(dataset_15min.iloc[:,m].values.tolist()))
#     _, p_30min = stats.normaltest(np.array(dataset_30min.iloc[:,m].values.tolist()))
#     _, p_45min = stats.normaltest(np.array(dataset_45min.iloc[:,m].values.tolist()))
#     _, p_60min = stats.normaltest(np.array(dataset_60min.iloc[:,m].values.tolist()))
#     print(p_15min);print(p_30min);print(p_45min);print(p_60min);

# print(stats.ttest_ind(np.array(dataset_15min.iloc[:, 3].values.tolist()), np.array(dataset_45min.iloc[:, 1].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_15min.iloc[:, 3].values.tolist()), np.array(dataset_15min.iloc[:, 2].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_15min.iloc[:, 3].values.tolist()), np.array(dataset_15min.iloc[:, 5].values.tolist())))


# m=3
# # # print(stats.ttest_ind(np.array(dataset_15min.iloc[:, m].values.tolist()), np.array(dataset_30min.iloc[:, m].values.tolist())))
# # # print(stats.ttest_ind(np.array(dataset_15min.iloc[:, m].values.tolist()), np.array(dataset_45min.iloc[:, m].values.tolist())))
# # # print(stats.ttest_ind(np.array(dataset_15min.iloc[:, m].values.tolist()), np.array(dataset_60min.iloc[:, m].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_30min.iloc[:, m].values.tolist()), np.array(dataset_45min.iloc[:, m].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_30min.iloc[:, m].values.tolist()), np.array(dataset_60min.iloc[:, m].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_45min.iloc[:, m].values.tolist()), np.array(dataset_60min.iloc[:, m].values.tolist())))



# for m in range(1,6):
#     print(np.median(dataset_15min_ECG.iloc[:,m]));print(np.median(dataset_30min_ECG.iloc[:,m]));
#     print(np.median(dataset_45min_ECG.iloc[:,m]));print(np.median(dataset_60min_ECG.iloc[:,m]));
# #     # _, p_15min = stats.normaltest(np.array(dataset_15min_ECG.iloc[:, m].values.tolist()))
# #     # _, p_30min = stats.normaltest(np.array(dataset_30min_ECG.iloc[:, m].values.tolist()))
# #     # _, p_45min = stats.normaltest(np.array(dataset_45min_ECG.iloc[:, m].values.tolist()))
# #     # _, p_60min = stats.normaltest(np.array(dataset_60min_ECG.iloc[:, m].values.tolist()))
# #     # print(p_15min);print(p_30min);print(p_45min);print(p_60min);
#
# # print(stats.ttest_ind(np.array(dataset_60min_ECG.iloc[:, 1].values.tolist()), np.array(dataset_60min_ECG.iloc[:, 2].values.tolist())))
# # print(stats.ttest_ind(np.array(dataset_60min_ECG.iloc[:, 1].values.tolist()), np.array(dataset_60min_ECG.iloc[:, 3].values.tolist())))
# # print(stats.ttest_ind(np.array(dataset_60min_ECG.iloc[:, 1].values.tolist()), np.array(dataset_45min_ECG.iloc[:, 5].values.tolist())))
#
# m=2
# print(stats.ttest_ind(np.array(dataset_60min_ECG.iloc[:, m].values.tolist()), np.array(dataset_45min_ECG.iloc[:, m].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_60min_ECG.iloc[:, m].values.tolist()), np.array(dataset_30min_ECG.iloc[:, m].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_60min_ECG.iloc[:, m].values.tolist()), np.array(dataset_15min_ECG.iloc[:, m].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_15min_ECG.iloc[:, m].values.tolist()), np.array(dataset_30min_ECG.iloc[:, m].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_15min_ECG.iloc[:, m].values.tolist()), np.array(dataset_45min_ECG.iloc[:, m].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_30min_ECG.iloc[:, m].values.tolist()), np.array(dataset_45min_ECG.iloc[:, m].values.tolist())))


# for m in range(1,6):
#     print(np.median(dataset_15min_EEGECG.iloc[:,m]));print(np.median(dataset_30min_EEGECG.iloc[:,m]));
#     print(np.median(dataset_45min_EEGECG.iloc[:,m]));print(np.median(dataset_60min_EEGECG.iloc[:,m]));
#     # _, p_15min = stats.normaltest(np.array(dataset_15min_EEGECG.iloc[:, m].values.tolist()))
#     # _, p_30min = stats.normaltest(np.array(dataset_30min_EEGECG.iloc[:, m].values.tolist()))
#     # _, p_45min = stats.normaltest(np.array(dataset_45min_EEGECG.iloc[:, m].values.tolist()))
#     # _, p_60min = stats.normaltest(np.array(dataset_60min_EEGECG.iloc[:, m].values.tolist()))
#     # print(p_15min);print(p_30min);print(p_45min);print(p_60min);
# m=3
# print(stats.ttest_ind(np.array(dataset_15min_EEGECG.iloc[:, m].values.tolist()), np.array(dataset_30min_EEGECG.iloc[:, m].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_15min_EEGECG.iloc[:, m].values.tolist()), np.array(dataset_45min_EEGECG.iloc[:, m].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_15min_EEGECG.iloc[:, m].values.tolist()), np.array(dataset_60min_EEGECG.iloc[:, m].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_30min_EEGECG.iloc[:, m].values.tolist()), np.array(dataset_45min_EEGECG.iloc[:, m].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_30min_EEGECG.iloc[:, m].values.tolist()), np.array(dataset_60min_EEGECG.iloc[:, m].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_45min_EEGECG.iloc[:, m].values.tolist()), np.array(dataset_60min_EEGECG.iloc[:, m].values.tolist())))



# m=3
# print(stats.ttest_ind(np.array(dataset_15min.iloc[:, 3].values.tolist()), np.array(dataset_15min_EEGECG.iloc[:, 3].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_30min.iloc[:, 3].values.tolist()), np.array(dataset_30min_EEGECG.iloc[:, 3].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_45min.iloc[:, 3].values.tolist()), np.array(dataset_45min_EEGECG.iloc[:, 3].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_60min.iloc[:, 3].values.tolist()), np.array(dataset_60min_EEGECG.iloc[:, 3].values.tolist())))




### plot p values
# import matplotlib.ticker as mticker
# f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
#
# data_15min_RFT=dataset_15min.iloc[:, 3]
# data_30min_RFT=dataset_30min.iloc[:, 3]
# data_45min_RFT=dataset_45min.iloc[:, 3]
# data_60min_RFT=dataset_60min.iloc[:, 3]
# # # _, p_15min = stats.normaltest(np.array(dataset_15min_statistic.iloc[:,3].values.tolist()))
# # # _, p_30min = stats.normaltest(np.array(dataset_30min_statistic.iloc[:,3].values.tolist()))
# # # _, p_45min = stats.normaltest(np.array(dataset_45min_statistic.iloc[:,3].values.tolist()))
# # # _, p_60min = stats.normaltest(np.array(dataset_60min_statistic.iloc[:,3].values.tolist()))
# # # print(p_15min);print(p_30min);print(p_45min);print(p_60min);
# #
# fig1, ax = pyplot.subplots(figsize=(5, 4))
# bp1 = ax.boxplot(data_15min_RFT, positions=[1], widths=0.3,patch_artist=True, showfliers=False)
# bp2 = ax.boxplot(data_30min_RFT, positions=[2], widths=0.3,patch_artist=True, showfliers=False)
# bp3 = ax.boxplot(data_45min_RFT , positions=[3], widths=0.3,patch_artist=True, showfliers=False)
# bp4 = ax.boxplot(data_60min_RFT, positions=[4], widths=0.3,patch_artist=True, showfliers=False)
# bp1['boxes'][0].set(color='k', linewidth=0.8)
# bp1['boxes'][0].set(facecolor='w')
# bp2['boxes'][0].set(color='k', linewidth=0.8)
# bp2['boxes'][0].set(facecolor='w')
# bp3['boxes'][0].set(color='k', linewidth=0.8)
# bp3['boxes'][0].set(facecolor='w')
# bp4['boxes'][0].set(color='k', linewidth=0.8)
# bp4['boxes'][0].set(facecolor='w')
#
# x1, x2 = 1, 2
# y, h, col = 0.89 + 0.001, 0.001, 'k'
# pyplot.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# # pyplot.text((x1+x2)*.5, 0.892, "p=2.5e-29", ha='center', va='bottom', color=col,fontsize=10)
# pyplot.text((x1+x2)*.5, 0.8908, "p=${}$".format(f._formatSciNotation('%1.10e' % 2.5e-29)), ha='center', va='bottom', color=col,fontsize=9)
#
# x1, x2 = 1, 3
# y, h, col = 0.89 + 0.007, 0.001, 'k'
# pyplot.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# # pyplot.text((x1+x2)*.5, 0.898, "p=7.9e-25", ha='center', va='bottom', color=col,fontsize=10)
# pyplot.text((x1+x2)*.5, 0.8968, "p=${}$".format(f._formatSciNotation('%1.10e' % 7.9e-25)), ha='center', va='bottom', color=col,fontsize=9)
#
# x1, x2 = 1, 4
# y, h, col = 0.89 + 0.013, 0.001, 'k'
# pyplot.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# # pyplot.text((x1+x2)*.5, 0.904, "p=8.3e-28", ha='center', va='bottom', color=col,fontsize=10)
# pyplot.text((x1+x2)*.5, 0.9036, "p=${}$".format(f._formatSciNotation('%1.10e' % 8.3e-28)), ha='center', va='bottom', color=col,fontsize=9)
#
#
#
# locs, labels = pyplot.yticks([0.80, 0.85, 0.90],['0.80','0.85','0.90'],fontsize=10)
# # locs, labels = pyplot.yticks([0.5,0.7, 0.9],['0.5','0.7','0.9'],fontsize=10)
# locs, labels = pyplot.xticks([1,2,3,4],['15-0 min','30-15 min','45-30 min','60-45 min'],fontsize=10)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Preictal periods',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
# # ax.set_title('Preictal data classification(EEG, RFT)',fontsize=12)
# pyplot.tight_layout()
# pyplot.show()
#
#
#
#
# data_15min_RFT_ECG=dataset_15min_ECG.iloc[:, 3]
# data_30min_RFT_ECG=dataset_30min_ECG.iloc[:, 3]
# data_45min_RFT_ECG=dataset_45min_ECG.iloc[:, 3]
# data_60min_RFT_ECG=dataset_60min_ECG.iloc[:, 3]
# # # _, p_15min = stats.normaltest(np.array(dataset_15min_statistic.iloc[:,3].values.tolist()))
# # # _, p_30min = stats.normaltest(np.array(dataset_30min_statistic.iloc[:,3].values.tolist()))
# # # _, p_45min = stats.normaltest(np.array(dataset_45min_statistic.iloc[:,3].values.tolist()))
# # # _, p_60min = stats.normaltest(np.array(dataset_60min_statistic.iloc[:,3].values.tolist()))
# # # print(p_15min);print(p_30min);print(p_45min);print(p_60min);
#
# fig1, ax = pyplot.subplots(figsize=(5, 4))
# bp1 = ax.boxplot(data_15min_RFT_ECG, positions=[1], widths=0.3,patch_artist=True, showfliers=False)
# bp2 = ax.boxplot(data_30min_RFT_ECG, positions=[2], widths=0.3,patch_artist=True, showfliers=False)
# bp3 = ax.boxplot(data_45min_RFT_ECG, positions=[3], widths=0.3,patch_artist=True, showfliers=False)
# bp4 = ax.boxplot(data_60min_RFT_ECG, positions=[4], widths=0.3,patch_artist=True, showfliers=False)
# bp1['boxes'][0].set(color='k', linewidth=0.8)
# bp1['boxes'][0].set(facecolor='w')
# bp2['boxes'][0].set(color='k', linewidth=0.8)
# bp2['boxes'][0].set(facecolor='w')
# bp3['boxes'][0].set(color='k', linewidth=0.8)
# bp3['boxes'][0].set(facecolor='w')
# bp4['boxes'][0].set(color='k', linewidth=0.8)
# bp4['boxes'][0].set(facecolor='w')
#
#
#
#
# x1, x2 = 4, 3
# y, h, col = 0.59 + 0.001, 0.001, 'k'
# pyplot.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# pyplot.text((x1+x2)*.5, 0.591, "p=${}$".format(f._formatSciNotation('%1.10e' % 1.6e-53)), ha='center', va='bottom', color=col,fontsize=9)
# x1, x2 = 4, 2
# y, h, col = 0.59 + 0.007, 0.001, 'k'
# pyplot.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# pyplot.text((x1+x2)*.5, 0.5965, "p=${}$".format(f._formatSciNotation('%1.10e' % 1.7e-06)), ha='center', va='bottom', color=col,fontsize=9)
# x1, x2 = 4, 1
# y, h, col = 0.59 + 0.013, 0.001, 'k'
# pyplot.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# pyplot.text((x1+x2)*.5, 0.6040, "p=${}$".format(f._formatSciNotation('%1.10e' % 2.3e-45)), ha='center', va='bottom', color=col,fontsize=9)
#
#
#
# locs, labels = pyplot.yticks([0.50,0.55, 0.60],['0.50','0.55','0.60'],fontsize=10)
# locs, labels = pyplot.xticks([1,2,3,4],['15-0 min','30-15 min','45-30 min','60-45 min'],fontsize=10)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Preictal periods',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
# pyplot.tight_layout()
# pyplot.show()
#
#
#
#
#
# data_15min_RFT_EEGECG=dataset_15min_EEGECG.iloc[:, 3]
# data_30min_RFT_EEGECG=dataset_30min_EEGECG.iloc[:, 3]
# data_45min_RFT_EEGECG=dataset_45min_EEGECG.iloc[:, 3]
# data_60min_RFT_EEGECG=dataset_60min_EEGECG.iloc[:, 3]
#
#
# fig1, ax = pyplot.subplots(figsize=(5, 4))
# bp1 = ax.boxplot(data_15min_RFT_EEGECG, positions=[1], widths=0.3,patch_artist=True, showfliers=False)
# bp2 = ax.boxplot(data_30min_RFT_EEGECG, positions=[2], widths=0.3,patch_artist=True, showfliers=False)
# bp3 = ax.boxplot(data_45min_RFT_EEGECG , positions=[3], widths=0.3,patch_artist=True, showfliers=False)
# bp4 = ax.boxplot(data_60min_RFT_EEGECG, positions=[4], widths=0.3,patch_artist=True, showfliers=False)
# bp1['boxes'][0].set(color='k', linewidth=0.8)
# bp1['boxes'][0].set(facecolor='w')
# bp2['boxes'][0].set(color='k', linewidth=0.8)
# bp2['boxes'][0].set(facecolor='w')
# bp3['boxes'][0].set(color='k', linewidth=0.8)
# bp3['boxes'][0].set(facecolor='w')
# bp4['boxes'][0].set(color='k', linewidth=0.8)
# bp4['boxes'][0].set(facecolor='w')
#
# x1, x2 = 1, 2
# y, h, col = 0.89 + 0.001, 0.001, 'k'
# pyplot.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# pyplot.text((x1+x2)*.5, 0.891, "p=${}$".format(f._formatSciNotation('%1.10e' % 2.4e-32)), ha='center', va='bottom', color=col,fontsize=9)
#
# x1, x2 = 1, 3
# y, h, col = 0.89 + 0.007, 0.001, 'k'
# pyplot.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# pyplot.text((x1+x2)*.5, 0.8968, "p=${}$".format(f._formatSciNotation('%1.10e' % 5.1e-23)), ha='center', va='bottom', color=col,fontsize=9)
#
# x1, x2 = 1, 4
# y, h, col = 0.89 + 0.013, 0.001, 'k'
# pyplot.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# pyplot.text((x1+x2)*.5, 0.903, "p=${}$".format(f._formatSciNotation('%1.10e' % 1.3e-7)), ha='center', va='bottom', color=col,fontsize=9)
#
#
# locs, labels = pyplot.yticks([0.80, 0.85, 0.90],['0.80','0.85','0.90'],fontsize=10)
# # locs, labels = pyplot.yticks([0.5,0.7, 0.9],['0.5','0.7','0.9'],fontsize=10)
# locs, labels = pyplot.xticks([1,2,3,4],['15-0 min','30-15 min','45-30 min','60-45 min'],fontsize=10)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Preictal periods',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
# # ax.set_title('Preictal data classification(EEGECG, RFT)',fontsize=12)
# pyplot.tight_layout()
# pyplot.show()



# dataset_15min=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_15_0min_channels_EEGperformance_Accuracy_x.csv',sep=',')
# dataset_30min=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_30_15min_channels_EEGperformance_Accuracy_x.csv',sep=',')
# dataset_45min=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_45_30min_channels_EEGperformance_Accuracy_x.csv',sep=',')
# dataset_60min=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_60_45min_channels_EEGperformance_Accuracy_x.csv',sep=',')
#
#
# dataset_15min_EEGECG=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_15_0min_channels_EEGECGperformance_Accuracy_x.csv',sep=',')
# dataset_30min_EEGECG=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_30_15min_channels_EEGECGperformance_Accuracy_x.csv',sep=',')
# dataset_45min_EEGECG=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_45_30min_channels_EEGECGperformance_Accuracy_x.csv',sep=',')
# dataset_60min_EEGECG=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_60_45min_channels_EEGECGperformance_Accuracy_x.csv',sep=',')
#
#
# dataset_15min_statistic=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/statistic_raw_15_0min_channels_EEGperformance_Accuracy_test.csv',sep=',')
# dataset_30min_statistic=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/statistic_raw_30_15min_channels_EEGperformance_Accuracy_test.csv',sep=',')
# dataset_45min_statistic=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/statistic_raw_45_30min_channels_EEGperformance_Accuracy_test.csv',sep=',')
# dataset_60min_statistic=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/statistic_raw_60_45min_channels_EEGperformance_Accuracy_test.csv',sep=',')
#
#
# dataset_15min_EEGECG_statistic=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/statistic_raw_15_0min_channels_EEGECGperformance_Accuracy.csv',sep=',')
# dataset_30min_EEGECG_statistic=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/statistic_raw_30_15min_channels_EEGECGperformance_Accuracy.csv',sep=',')
# dataset_45min_EEGECG_statistic=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/statistic_raw_45_30min_channels_EEGECGperformance_Accuracy.csv',sep=',')
# dataset_60min_EEGECG_statistic=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/statistic_raw_60_45min_channels_EEGECGperformance_Accuracy.csv',sep=',')
#
#
# data_15min_statistic=dataset_15min_statistic.iloc[:, 3]
# data_30min_statistic=dataset_30min_statistic.iloc[:, 3]
# data_45min_statistic=dataset_45min_statistic.iloc[:, 3]
# data_60min_statistic=dataset_60min_statistic.iloc[:, 3]
# data_15min_RFT=dataset_15min.iloc[:, 3]
# data_30min_RFT=dataset_30min.iloc[:, 3]
# data_45min_RFT=dataset_45min.iloc[:, 3]
# data_60min_RFT=dataset_60min.iloc[:, 3]
#
# print(np.mean(data_15min_RFT));print(np.mean(data_15min_statistic));
# print(np.mean(data_30min_RFT));print(np.mean(data_30min_statistic));
# print(np.mean(data_45min_RFT));print(np.mean(data_45min_statistic));
# print(np.mean(data_60min_RFT));print(np.mean(data_60min_statistic));
#
# print(np.std(data_15min_RFT));print(np.std(data_15min_statistic));
# print(np.std(data_30min_RFT));print(np.std(data_30min_statistic));
# print(np.std(data_45min_RFT));print(np.std(data_45min_statistic));
# print(np.std(data_60min_RFT));print(np.std(data_60min_statistic));
#
#
# print(stats.ttest_ind(np.array(data_15min_RFT.values.tolist()), np.array(data_15min_statistic.values.tolist())))
# print(stats.ttest_ind(np.array(data_30min_RFT.values.tolist()), np.array(data_30min_statistic.values.tolist())))
# print(stats.ttest_ind(np.array(data_45min_RFT.values.tolist()), np.array(data_45min_statistic.values.tolist())))
# print(stats.ttest_ind(np.array(data_60min_RFT.values.tolist()), np.array(data_60min_statistic.values.tolist())))
#
#
# fig1, ax = pyplot.subplots(figsize=(5, 4))
# bp1 = ax.boxplot(data_15min_RFT, positions=[1], widths=0.3,patch_artist=True, showfliers=False)
# bp2 = ax.boxplot(data_30min_RFT, positions=[3], widths=0.3,patch_artist=True, showfliers=False)
# bp3 = ax.boxplot(data_45min_RFT , positions=[5], widths=0.3,patch_artist=True, showfliers=False)
# bp4 = ax.boxplot(data_60min_RFT, positions=[7], widths=0.3,patch_artist=True, showfliers=False)
# bp5 = ax.boxplot(data_15min_statistic, positions=[2], widths=0.3,patch_artist=True, showfliers=False)
# bp6 = ax.boxplot(data_30min_statistic, positions=[4], widths=0.3,patch_artist=True, showfliers=False)
# bp7 = ax.boxplot(data_45min_statistic , positions=[6], widths=0.3,patch_artist=True, showfliers=False)
# bp8 = ax.boxplot(data_60min_statistic, positions=[8], widths=0.3,patch_artist=True, showfliers=False)
#
# bp1['boxes'][0].set(color='k', linewidth=0.8)
# bp1['boxes'][0].set(facecolor='g')
# bp2['boxes'][0].set(color='k', linewidth=0.8)
# bp2['boxes'][0].set(facecolor='g')
# bp3['boxes'][0].set(color='k', linewidth=0.8)
# bp3['boxes'][0].set(facecolor='g')
# bp4['boxes'][0].set(color='k', linewidth=0.8)
# bp4['boxes'][0].set(facecolor='g')
# # locs, labels = pyplot.yticks([0.85,0.87, 0.89],['0.85','0.87','0.89'],fontsize=10)
# locs, labels = pyplot.xticks([1,3,5,7],['15-0 min','30-15 min','45-30 min','60-45 min'],fontsize=10)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Preictal periods',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
# # ax.set_title('Preictal data classification(EEG, RFT)',fontsize=12)
# pyplot.tight_layout()
# pyplot.show()













# data_15min_statistic_EEGECG=dataset_15min_EEGECG_statistic.iloc[:, 3]
# data_30min_statistic_EEGECG=dataset_30min_EEGECG_statistic.iloc[:, 3]
# data_45min_statistic_EEGECG=dataset_45min_EEGECG_statistic.iloc[:, 3]
# data_60min_statistic_EEGECG=dataset_60min_EEGECG_statistic.iloc[:, 3]
# data_15min_RFT_EEGECG=dataset_15min_EEGECG.iloc[:, 3]
# data_30min_RFT_EEGECG=dataset_30min_EEGECG.iloc[:, 3]
# data_45min_RFT_EEGECG=dataset_45min_EEGECG.iloc[:, 3]
# data_60min_RFT_EEGECG=dataset_60min_EEGECG.iloc[:, 3]
#
#
# fig1, ax = pyplot.subplots(figsize=(5, 4))
# bp1 = ax.boxplot(data_15min_RFT_EEGECG, positions=[1], widths=0.3,patch_artist=True, showfliers=False)
# bp2 = ax.boxplot(data_30min_RFT_EEGECG, positions=[3], widths=0.3,patch_artist=True, showfliers=False)
# bp3 = ax.boxplot(data_45min_RFT_EEGECG , positions=[5], widths=0.3,patch_artist=True, showfliers=False)
# bp4 = ax.boxplot(data_60min_RFT_EEGECG, positions=[7], widths=0.3,patch_artist=True, showfliers=False)
# bp5 = ax.boxplot(data_15min_statistic_EEGECG, positions=[2], widths=0.3,patch_artist=True, showfliers=False)
# bp6 = ax.boxplot(data_30min_statistic_EEGECG, positions=[4], widths=0.3,patch_artist=True, showfliers=False)
# bp7 = ax.boxplot(data_45min_statistic_EEGECG , positions=[6], widths=0.3,patch_artist=True, showfliers=False)
# bp8 = ax.boxplot(data_60min_statistic_EEGECG, positions=[8], widths=0.3,patch_artist=True, showfliers=False)
#
# bp1['boxes'][0].set(color='k', linewidth=0.8)
# bp1['boxes'][0].set(facecolor='w')
# bp2['boxes'][0].set(color='k', linewidth=0.8)
# bp2['boxes'][0].set(facecolor='w')
# bp3['boxes'][0].set(color='k', linewidth=0.8)
# bp3['boxes'][0].set(facecolor='w')
# bp4['boxes'][0].set(color='k', linewidth=0.8)
# bp4['boxes'][0].set(facecolor='w')
# locs, labels = pyplot.yticks([0.85,0.87, 0.89],['0.85','0.87','0.89'],fontsize=10)
# locs, labels = pyplot.xticks([1,3,5,7],['15-0 min','30-15 min','45-30 min','60-45 min'],fontsize=10)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Preictal periods',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
# # ax.set_title('Preictal data classification(EEG, RFT)',fontsize=12)
# pyplot.tight_layout()
# pyplot.show()
#
#
# print(stats.ttest_ind(np.array(data_15min_RFT_EEGECG.values.tolist()), np.array(data_15min_statistic_EEGECG.values.tolist())))
# print(stats.ttest_ind(np.array(data_30min_RFT_EEGECG.values.tolist()), np.array(data_30min_statistic_EEGECG.values.tolist())))
# print(stats.ttest_ind(np.array(data_45min_RFT_EEGECG.values.tolist()), np.array(data_45min_statistic_EEGECG.values.tolist())))
# print(stats.ttest_ind(np.array(data_60min_RFT_EEGECG.values.tolist()), np.array(data_60min_statistic_EEGECG.values.tolist())))







# data_15min_EEGECG = [dataset_15min.iloc[:, 1],dataset_15min_ECG.iloc[:, 1],dataset_15min_EEGECG.iloc[:, 1],
#                      dataset_15min.iloc[:, 2],dataset_15min_ECG.iloc[:, 2],dataset_15min_EEGECG.iloc[:, 2],
#                      dataset_15min.iloc[:, 3],dataset_15min_ECG.iloc[:, 3],dataset_15min_EEGECG.iloc[:, 3],
#                      dataset_15min.iloc[:, 4],dataset_15min_ECG.iloc[:, 4],dataset_15min_EEGECG.iloc[:, 4],
#                      dataset_15min.iloc[:, 5],dataset_15min_ECG.iloc[:, 5],dataset_15min_EEGECG.iloc[:, 5]]
#
# data_30min_EEGECG = [dataset_30min.iloc[:, 1],dataset_30min_ECG.iloc[:, 1],dataset_30min_EEGECG.iloc[:, 1],
#                      dataset_30min.iloc[:, 2],dataset_30min_ECG.iloc[:, 2],dataset_30min_EEGECG.iloc[:, 2],
#                      dataset_30min.iloc[:, 3],dataset_30min_ECG.iloc[:, 3],dataset_30min_EEGECG.iloc[:, 3],
#                      dataset_30min.iloc[:, 4],dataset_30min_ECG.iloc[:, 4],dataset_30min_EEGECG.iloc[:, 4],
#                      dataset_30min.iloc[:, 5],dataset_30min_ECG.iloc[:, 5],dataset_30min_EEGECG.iloc[:, 5]]
#
# data_45min_EEGECG = [dataset_45min.iloc[:, 1],dataset_45min_ECG.iloc[:, 1],dataset_45min_EEGECG.iloc[:, 1],
#                      dataset_45min.iloc[:, 2],dataset_45min_ECG.iloc[:, 2],dataset_45min_EEGECG.iloc[:, 2],
#                      dataset_45min.iloc[:, 3],dataset_45min_ECG.iloc[:, 3],dataset_45min_EEGECG.iloc[:, 3],
#                      dataset_45min.iloc[:, 4],dataset_45min_ECG.iloc[:, 5],dataset_45min_EEGECG.iloc[:, 5],
#                      dataset_45min.iloc[:, 5],dataset_45min_ECG.iloc[:, 5],dataset_45min_EEGECG.iloc[:, 5]]
#
# data_60min_EEGECG = [dataset_60min.iloc[:, 1],dataset_60min_ECG.iloc[:, 1],dataset_60min_EEGECG.iloc[:, 1],
#                      dataset_60min.iloc[:, 2],dataset_60min_ECG.iloc[:, 2],dataset_60min_EEGECG.iloc[:, 2],
#                      dataset_60min.iloc[:, 3],dataset_60min_ECG.iloc[:, 3],dataset_60min_EEGECG.iloc[:, 3],
#                      dataset_60min.iloc[:, 4],dataset_60min_ECG.iloc[:, 4],dataset_60min_EEGECG.iloc[:, 4],
#                      dataset_60min.iloc[:, 5],dataset_60min_ECG.iloc[:, 5],dataset_60min_EEGECG.iloc[:, 5]]
#
# fig1, ax = pyplot.subplots(figsize=(8, 4))
# bp1 = ax.boxplot(data_15min_EEGECG, positions=[1,2,3,4,5,6,7,8,9,10], widths=0.4,patch_artist=True, showfliers=False)
# bp2 = ax.boxplot(data_30min_EEGECG, positions=[12,13,14,15,16,17,18,19,20,21], widths=0.4,patch_artist=True, showfliers=False)
# bp3 = ax.boxplot(data_45min_EEGECG , positions=[23,24,25,26,27,28,29,30,31,32], widths=0.4,patch_artist=True, showfliers=False)
# bp4 = ax.boxplot(data_60min_EEGECG, positions=[34,35,36,37,38,39,40,41,42,43], widths=0.4,patch_artist=True, showfliers=False)
#
#
# for box in bp1['boxes']:
#     if box == bp1['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp1['boxes'][1]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp1['boxes'][2]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp1['boxes'][3]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp1['boxes'][4]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp1['boxes'][5]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp1['boxes'][6]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp1['boxes'][7]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp1['boxes'][8]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
#     if box == bp1['boxes'][9]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='w')
#
# for box in bp2['boxes']:
#     if box == bp2['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp2['boxes'][1]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp2['boxes'][2]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp2['boxes'][3]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp2['boxes'][4]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp2['boxes'][5]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp2['boxes'][6]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp2['boxes'][7]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp2['boxes'][8]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
#     if box == bp2['boxes'][9]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='w')
#
# for box in bp3['boxes']:
#     if box == bp3['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp3['boxes'][1]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp3['boxes'][2]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp3['boxes'][3]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp3['boxes'][4]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp3['boxes'][5]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp3['boxes'][6]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp3['boxes'][7]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp3['boxes'][8]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
#     if box == bp3['boxes'][9]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='w')
#
# for box in bp4['boxes']:
#     if box == bp4['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp4['boxes'][1]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp4['boxes'][2]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp4['boxes'][3]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp4['boxes'][4]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp4['boxes'][5]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp4['boxes'][6]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp4['boxes'][7]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp4['boxes'][8]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
#     if box == bp4['boxes'][9]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='w')
#
# locs, labels = pyplot.yticks([0.6,0.7,0.8],['0.6','0.7','0.8'],fontsize=12)
# locs, labels = pyplot.xticks([6,16,27,38],['15-0 min','30-15 min','45-30 min','60-45 min'],fontsize=10)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Preictal periods',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
# ax.set_title('Preictal data classification',fontsize=12)
# ax.legend([bp1["boxes"][0],bp1["boxes"][2], bp1["boxes"][4],bp1["boxes"][6],bp1["boxes"][8]], ['KNN','DT','RFT','NB','SVM'], loc='lower left',fontsize=7)
# pyplot.tight_layout()
# pyplot.show()
#
# print(np.mean(dataset_15min_EEGECG.iloc[:,3]));print(np.mean(dataset_30min_EEGECG.iloc[:,3]));print(np.mean(dataset_45min_EEGECG.iloc[:,3]));print(np.mean(dataset_60min_EEGECG.iloc[:,3]));
# print(np.std(dataset_15min_EEGECG.iloc[:,3]));print(np.std(dataset_30min_EEGECG.iloc[:,3]));print(np.std(dataset_45min_EEGECG.iloc[:,3]));print(np.std(dataset_60min_EEGECG.iloc[:,3]));
#
# _, p_15min = stats.normaltest(np.array(dataset_15min_EEGECG.iloc[:,3].values.tolist()))
# _, p_30min = stats.normaltest(np.array(dataset_30min_EEGECG.iloc[:,3].values.tolist()))
# _, p_45min = stats.normaltest(np.array(dataset_45min_EEGECG.iloc[:,3].values.tolist()))
# _, p_60min = stats.normaltest(np.array(dataset_60min_EEGECG.iloc[:,3].values.tolist()))
# print(p_15min);print(p_30min);print(p_45min);print(p_60min);
#
# for m in range(1,6):
#     print(np.mean(dataset_15min_EEGECG.iloc[:,m]));print(np.mean(dataset_30min_EEGECG.iloc[:,m]));
#     print(np.mean(dataset_45min_EEGECG.iloc[:,m]));print(np.mean(dataset_60min_EEGECG.iloc[:,m]));
#
# print(stats.ttest_ind(np.array(dataset_15min_EEGECG.iloc[:, 5].values.tolist()), np.array(dataset_30min_EEGECG.iloc[:, 5].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_15min_EEGECG.iloc[:, 5].values.tolist()), np.array(dataset_45min_EEGECG.iloc[:, 5].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_15min_EEGECG.iloc[:, 5].values.tolist()), np.array(dataset_60min_EEGECG.iloc[:, 5].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_30min_EEGECG.iloc[:, 5].values.tolist()), np.array(dataset_45min_EEGECG.iloc[:, 5].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_30min_EEGECG.iloc[:, 5].values.tolist()), np.array(dataset_60min_EEGECG.iloc[:, 5].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_45min_EEGECG.iloc[:, 5].values.tolist()), np.array(dataset_60min_EEGECG.iloc[:, 5].values.tolist())))


# print(stats.ttest_ind(np.array(dataset_15min.iloc[:, 5].values.tolist()), np.array(dataset_15min.iloc[:,2].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_30min.iloc[:, 5].values.tolist()), np.array(dataset_30min.iloc[:,2].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_45min.iloc[:, 5].values.tolist()), np.array(dataset_45min.iloc[:,2].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_60min.iloc[:, 5].values.tolist()), np.array(dataset_60min.iloc[:,2].values.tolist())))

# print(stats.ttest_ind(np.array(dataset_15min.iloc[:, 3].values.tolist()), np.array(dataset_15min.iloc[:,1].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_15min.iloc[:, 3].values.tolist()), np.array(dataset_30min.iloc[:,2].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_15min.iloc[:, 3].values.tolist()), np.array(dataset_15min.iloc[:,4].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_15min.iloc[:, 3].values.tolist()), np.array(dataset_60min.iloc[:,5].values.tolist())))


# print(stats.ttest_ind(np.array(dataset_15min.iloc[:, 5].values.tolist()), np.array(dataset_15min_EEGECG.iloc[:, 5].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_30min.iloc[:, 5].values.tolist()), np.array(dataset_30min_EEGECG.iloc[:, 5].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_45min.iloc[:, 5].values.tolist()), np.array(dataset_45min_EEGECG.iloc[:, 5].values.tolist())))
# print(stats.ttest_ind(np.array(dataset_60min.iloc[:, 5].values.tolist()), np.array(dataset_60min_EEGECG.iloc[:, 5].values.tolist())))


# dataset_15min_CNN=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/CNN/CS_CNN_15min_ALL_EEGperformance_Accuracy.csv',sep=',')
# dataset_30min_CNN=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/CNN/CS_CNN_30min_ALL_EEGperformance_Accuracy.csv',sep=',')
# dataset_15min_EEGECG_CNN=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/CNN/CS_CNN_15min_ALL_EEGECGperformance_Accuracy.csv',sep=',')
# dataset_30min_EEGECG_CNN=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/CNN/CS_CNN_30min_ALL_EEGECGperformance_Accuracy.csv',sep=',')
#
# print(np.mean(dataset_15min_CNN.iloc[:, 1]))
# print(np.mean(dataset_30min_CNN.iloc[:, 1]))
#
# dataset_15min_30min_CNN=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/CNN/CS_CNN_15min_30min_ALL_EEGperformance_Accuracy.csv',sep=',')
#
# fig1, ax = pyplot.subplots()
# data_sum=[dataset_15min_CNN.iloc[:, 1],dataset_15min_EEGECG_CNN.iloc[:, 1],dataset_30min_CNN.iloc[:, 1],dataset_30min_EEGECG_CNN.iloc[:, 1]]
# bp1 = ax.boxplot(data_sum, positions=[1,2,3,4], widths=0.4,patch_artist=True, showfliers=False)
# for box in bp1['boxes']:
#     if box == bp1['boxes'][0]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
#     if box == bp1['boxes'][1]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='w')
#     if box == bp1['boxes'][2]:
#         box.set(color='tomato', linewidth=0.8)
#         box.set(facecolor='tomato')
#     if box == bp1['boxes'][3]:
#         box.set(color='tomato', linewidth=0.8)
#         box.set(facecolor='w')
# locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
#
# locs, labels = pyplot.xticks([1,2,3,4],['15-0 min','','30-15 min',''],fontsize=10)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Preictal periods',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
# ax.set_title('Preictal data classification (CNN)',fontsize=12)
# # ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3]], ['15-0 min','30-15 min','45-30 min','60-45 min'], loc='lower right',fontsize=7)
# pyplot.tight_layout()
# pyplot.show()
#
#
# fig1, ax = pyplot.subplots()
# data_sum=[dataset_15min_CNN.iloc[:, 1],dataset_30min_CNN.iloc[:, 1],dataset_15min_30min_CNN.iloc[:, 1]]
# bp1 = ax.boxplot(data_sum, positions=[1,2,3], widths=0.4,patch_artist=True, showfliers=False)
# for box in bp1['boxes']:
#     if box == bp1['boxes'][0]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
#     if box == bp1['boxes'][1]:
#         box.set(color='tomato', linewidth=0.8)
#         box.set(facecolor='tomato')
#     if box == bp1['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
# locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
#
# locs, labels = pyplot.xticks([1,2,3,4],['15-0 min','30-15 min','30-0 min'],fontsize=10)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Preictal periods',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
# ax.set_title('Preictal data classification (CNN)',fontsize=12)
# # ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3]], ['15-0 min','30-15 min','45-30 min','60-45 min'], loc='lower right',fontsize=7)
# pyplot.tight_layout()
# pyplot.show()





# # plot figures
# fig1, ax = pyplot.subplots(1,3, figsize=(8, 3))
#
# custom_ylim = (0.5, 0.9)
# pyplot.setp(ax, ylim=custom_ylim)
# for axs in ax.flat:
#     axs.spines['left'].set_visible(False)
#     axs.spines['right'].set_visible(False)
#     axs.spines['top'].set_visible(False)
#     axs.spines['bottom'].set_visible(False)
#     axs.set_yticks([])
#     axs.set_xticks([])
#
#
# bp1 = ax[0].boxplot(data_15min, positions=[1,1.5,2,2.5,3], widths=0.5,patch_artist=True, showfliers=False)
# bp2 = ax[0].boxplot(data_30min, positions=[6,6.5,7,7.5,8], widths=0.5,patch_artist=True, showfliers=False)
# bp3 = ax[0].boxplot(data_45min , positions=[11,11.5,12,12.5,13], widths=0.5,patch_artist=True, showfliers=False)
# bp4 = ax[0].boxplot(data_60min, positions=[16,16.5,17,17.5,18], widths=0.5,patch_artist=True, showfliers=False)
# for box in bp1['boxes']:
#     if box == bp1['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp1['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp1['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp1['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp1['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp3['boxes']:
#     if box == bp3['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp3['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp3['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp3['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp3['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp4['boxes']:
#     if box == bp4['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp4['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp4['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp4['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp4['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp2['boxes']:
#     if box == bp2['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp2['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp2['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp2['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp2['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# # locs, labels = pyplot.yticks([0.6,0.7,0.8],['0.6','0.7','0.8'],fontsize=12)
# ax[0].set_yticks([0.5,0.7,0.9])
# ax[0].set_yticklabels(['0.5','0.7','0.9'],fontsize=12)
# ax[0].set_xticks([2,7,12,17])
# ax[0].set_xticklabels(['15-0 min','30-15 min','45-30 min','60-45 min'],rotation = 13,fontsize=10)
# ax[0].spines['right'].set_visible(False)
# ax[0].spines['top'].set_visible(False)
# ax[0].set_xlabel('Preictal periods',fontsize=13)
# ax[0].set_ylabel('Accuracy',fontsize=13)
# ax[0].set_title('EEG',fontsize=12)
# # ax[0].legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3],bp1["boxes"][4]], ['KNN','DT','RFT','NB','SVM'], loc='lower left',fontsize=6)
# # pyplot.tight_layout()
# # pyplot.show()
#
#
#
# bp1 = ax[1].boxplot(data_15min_ECG, positions=[1,1.5,2,2.5,3], widths=0.5,patch_artist=True, showfliers=False)
# bp2 = ax[1].boxplot(data_30min_ECG, positions=[6,6.5,7,7.5,8], widths=0.5,patch_artist=True, showfliers=False)
# bp3 = ax[1].boxplot(data_45min_ECG , positions=[11,11.5,12,12.5,13], widths=0.5,patch_artist=True, showfliers=False)
# bp4 = ax[1].boxplot(data_60min_ECG, positions=[16,16.5,17,17.5,18], widths=0.5,patch_artist=True, showfliers=False)
# for box in bp1['boxes']:
#     if box == bp1['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp1['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp1['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp1['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp1['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp3['boxes']:
#     if box == bp3['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp3['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp3['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp3['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp3['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp4['boxes']:
#     if box == bp4['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp4['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp4['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp4['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp4['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp2['boxes']:
#     if box == bp2['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp2['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp2['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp2['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp2['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# # axs[1].set_yticks([0.4,0.5,0.6])
# # axs[1].set_yticklabels(['0.4','0.5','0.6'],fontsize=12)
# # ax[1].set_yticks([0.5,0.7,0.9])
# # ax[1].set_yticklabels(['0.5','0.7','0.9'],fontsize=12)
# # axs[1].set_xticks([3,9,15,21])
# # axs[1].set_xticklabels(['15-0 min','30-15 min','45-30 min','60-45 min'],rotation = 15,fontsize=12)
#
# ax[1].spines['right'].set_visible(False)
# ax[1].spines['top'].set_visible(False)
# ax[1].spines['bottom'].set_visible(False)
# # axs[1].set_xlabel('Preictal periods',fontsize=12)
# # axs[1].set_ylabel('Accuracy',fontsize=12)
# ax[1].set_title('ECG',fontsize=12)
# ax[1].set_xticks([])
# ax[1].legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3],bp1["boxes"][4]], ['KNN','DT','RF','NB','SVM'], loc='center left',fontsize=6)
# # pyplot.tight_layout()
# # pyplot.show()
#
#
# bp1 = ax[2].boxplot(data_15min_EEGECG, positions=[1,1.5,2,2.5,3], widths=0.5,patch_artist=True, showfliers=False)
# bp2 = ax[2].boxplot(data_30min_EEGECG, positions=[6,6.5,7,7.5,8], widths=0.5,patch_artist=True, showfliers=False)
# bp3 = ax[2].boxplot(data_45min_EEGECG, positions=[11,11.5,12,12.5,13], widths=0.5,patch_artist=True, showfliers=False)
# bp4 = ax[2].boxplot(data_60min_EEGECG, positions=[16,16.5,17,17.5,18], widths=0.5,patch_artist=True, showfliers=False)
# for box in bp1['boxes']:
#     if box == bp1['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp1['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp1['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp1['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp1['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp3['boxes']:
#     if box == bp3['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp3['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp3['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp3['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp3['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp4['boxes']:
#     if box == bp4['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp4['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp4['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp4['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp4['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# for box in bp2['boxes']:
#     if box == bp2['boxes'][0]:
#         box.set(color='k', linewidth=0.8)
#         box.set(facecolor='k')
#     if box == bp2['boxes'][1]:
#         box.set(color='r', linewidth=0.8)
#         box.set(facecolor='r')
#     if box == bp2['boxes'][2]:
#         box.set(color='g', linewidth=0.8)
#         box.set(facecolor='g')
#     if box == bp2['boxes'][3]:
#         box.set(color='grey', linewidth=0.8)
#         box.set(facecolor='grey')
#     if box == bp2['boxes'][4]:
#         box.set(color='C0', linewidth=0.8)
#         box.set(facecolor='C0')
# # locs, labels = pyplot.yticks([0.6,0.7,0.8],['0.6','0.7','0.8'],fontsize=12)
# # ax[2].set_yticks([0.5,0.7,0.9])
# # ax[2].set_yticklabels(['0.5','0.7','0.9'],fontsize=12)
# # axs[2].set_xticks([3,9,15,21])
# # axs[2].set_xticklabels(['15-0 min','30-15 min','45-30 min','60-45 min'],rotation = 15,fontsize=12)
# ax[2].spines['right'].set_visible(False)
# ax[2].spines['top'].set_visible(False)
# ax[2].spines['bottom'].set_visible(False)
# ax[2].set_xticks([])
# # axs[2].set_xlabel('Preictal periods',fontsize=12)
# # axs[2].set_ylabel('Accuracy',fontsize=12)
# ax[2].set_title('EEGECG',fontsize=12)
# # ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3],bp1["boxes"][4]], ['KNN','DT','RFT','NB','SVM'], loc='center left',fontsize=7)
# pyplot.tight_layout()
# pyplot.show()



## different parameters

dataset_15min_EEGECG_5_100_rbf=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_15_0min_channels_EEGECGperformance_Accuracy_KNN5_SVMrbf_RF100.csv',sep=',')
acc_KNN5 = dataset_15min_EEGECG_5_100_rbf.iloc[:, 1]
acc_DT100 = dataset_15min_EEGECG_5_100_rbf.iloc[:, 3]

dataset_15min_EEGECG_10_200_sigmoid=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_15_0min_channels_EEGECGperformance_Accuracy_KNN10_SVMsigmoid_RF200.csv',sep=',')
acc_KNN10 = dataset_15min_EEGECG_10_200_sigmoid.iloc[:, 1]
acc_DT200 = dataset_15min_EEGECG_10_200_sigmoid.iloc[:, 3]

dataset_15min_EEGECG_3_50_rbf=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/performances/Preictal_classify/allchannels_June/raw_15_0min_channels_EEGECGperformance_Accuracy_KNN3_SVMrbf_RF50.csv',sep=',')
acc_KNN3 = dataset_15min_EEGECG_3_50_rbf.iloc[:, 1]
acc_DT50 = dataset_15min_EEGECG_3_50_rbf.iloc[:, 3]


_, p_15min = stats.normaltest(acc_KNN3)
_, p_30min = stats.normaltest(acc_KNN5)
_, p_45min = stats.normaltest(acc_KNN10)
print(p_15min);print(p_30min);print(p_45min);
print(np.median(acc_KNN3));print(np.median(acc_KNN5));print(np.median(acc_KNN10));
print(stats.ttest_ind(acc_KNN3, acc_KNN5))
print(stats.ttest_ind(acc_KNN10, acc_KNN5))


_, p_15min = stats.normaltest(acc_DT50)
_, p_30min = stats.normaltest(acc_DT100)
_, p_45min = stats.normaltest(acc_DT200)
print(p_15min);print(p_30min);print(p_45min);
print(np.median(acc_DT50));print(np.median(acc_DT100));print(np.median(acc_DT200));
print(stats.ttest_ind(acc_DT50, acc_DT100))
print(stats.ttest_ind(acc_DT200, acc_DT100))
