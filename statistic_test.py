from __future__ import division
import pandas as pd
import numpy as np
from matplotlib import pyplot
import scipy.stats as stats


import json


# ### show results
# fig1, ax = pyplot.subplots(figsize=(10, 4))
# import os
#
# data_15min=[]
# directory =r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\backup\preES_prePNES\15min\Accuracy'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# i=0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
#         print(entry.path)
#         data_15min.append(np.array(dataset.iloc[:,5]))
#
# data_30min=[]
# directory = r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\backup\preES_prePNES\30min\Accuracy'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d: d.path)
# i = 0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',', skipinitialspace=True)
#         print(entry.path)
#         data_30min.append(np.array(dataset.iloc[:, 5]))
#
# data_45min=[]
# directory = r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\backup\preES_prePNES\45min\Accuracy'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d: d.path)
# i = 0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',', skipinitialspace=True)
#         print(entry.path)
#         data_45min.append(np.array(dataset.iloc[:, 2]))
#
#
# data_60min=[]
# directory = r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\backup\preES_prePNES\60min\Accuracy'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d: d.path)
# i = 0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',', skipinitialspace=True)
#         print(entry.path)
#         data_60min.append(np.array(dataset.iloc[:, 2]))
#
# print(data_60min); print(len(data_60min))
# m=0
# for i in range(len(data_60min)):
#     data=[data_15min[i],data_30min[i],data_45min[i],data_60min[i]]
#
#     bp1 = ax.boxplot(data, positions=[1+m, 4+m, 7+m, 10+m], widths=2,
#                                      patch_artist=True, showfliers=False)
#     for box in bp1['boxes']:
#         if box == bp1['boxes'][0]:
#             box.set(color='k', linewidth=0.8)
#             box.set(facecolor='k')
#         if box == bp1['boxes'][1]:
#             box.set(color='r', linewidth=0.8)
#             box.set(facecolor='r')
#         if box == bp1['boxes'][2]:
#             box.set(color='g', linewidth=0.8)
#             box.set(facecolor='g')
#
#         if box == bp1['boxes'][3]:
#             box.set(color='darkblue', linewidth=0.8)
#             box.set(facecolor='darkblue')
#
#     m = m + 15
#
#
# locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
# # locs, labels = pyplot.yticks([0.6,0.7,0.8],['0.6','0.7','0.8'],fontsize=12)
# locs, labels = pyplot.xticks([5,20,35,50,65,80,95,110,125,140,155,170,185,200,215,230,245,260],['Fz','F3','F4', 'F7', 'F8', 'C4', 'C3', 'Cz', 'Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2'],fontsize=10)
# # ax.hlines(0.5,0.5,24.5,'k',linestyles='--',alpha=0.5)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Channels',fontsize=12)
# ax.set_ylabel('Accuracy',fontsize=12)
# # ax.set_ylabel('AUC',fontsize=12)
# ax.set_title('Preictal epileptic seizures and preictal PNES classification',fontsize=12)
# # ax.set_title('Preictal epileptic seizures and interictal PNES classification',fontsize=12)
# # ax.set_title('Preictal epileptic seizures and ictal PNES classification',fontsize=12)
# ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3]], ['15min','30min', '45min','60min'], loc='lower right',fontsize=7)
# pyplot.tight_layout()
# pyplot.show()



# ### show results
# # fig1, ax = pyplot.subplots(figsize=(10, 4))
# import os
#
# data_15min=[]
# directory =r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\backup\preES_prePNES\15min\Accuracy'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# i=0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
#         print(entry.path)
#         # data_15min=data_15min+ list(dataset.iloc[:,5])
#         data_15min.append(dataset.iloc[:,5].values)
#
# data_30min=[]
# directory = r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\backup\preES_prePNES\30min\Accuracy'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d: d.path)
# i = 0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',', skipinitialspace=True)
#         print(entry.path)
#         # data_30min=data_30min+ list(dataset.iloc[:, 5])
#         data_30min.append(dataset.iloc[:, 5].values)
# data_45min=[]
# directory = r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\backup\preES_prePNES\45min\Accuracy'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d: d.path)
# i = 0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',', skipinitialspace=True)
#         print(entry.path)
#         # data_45min=data_45min+ list(dataset.iloc[:, 2])
#         data_45min.append(dataset.iloc[:, 2].values)
# data_60min=[]
# directory = r'C:\Users\wxiong\Documents\PHD\combine_features\performances\Preictal_classify\backup\preES_prePNES\60min\Accuracy'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d: d.path)
# i = 0
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset = pd.read_csv(entry.path, sep=',', skipinitialspace=True)
#         print(entry.path)
#         # data_60min=data_60min+ list(dataset.iloc[:, 2])
#         data_60min.append(dataset.iloc[:, 2].values)
#
# print(data_60min); print(len(data_60min));print(type(data_60min));print(type(data_60min[0]));

# for i in range(len(data_15min)):
#     print(stats.ttest_ind(np.array(data_45min[i]), np.array(data_60min[i])))



# _, p_15min = stats.normaltest(np.array(data_15min))
# print(p_15min);
# _, p_30min = stats.normaltest(np.array(data_30min))
# print(p_30min);
# _, p_45min = stats.normaltest(np.array(data_45min))
# print(p_45min);
# _, p_60min = stats.normaltest(np.array(data_60min))
# print(p_60min)

# fig1, axs = pyplot.subplots(2,2,sharex=True,sharey=True)
# fig1.suptitle('Accuracy distribution')
# axs[0, 0].hist(data_15min, 10,label='15min-0min')
# axs[0, 0].legend(loc='upper left',fontsize=8)
# axs[0, 0].spines['right'].set_visible(False)
# axs[0, 0].spines['top'].set_visible(False)
# axs[0, 1].hist(data_30min, 10,label='30min-15min')
# axs[0, 1].legend(loc='upper left',fontsize=8)
# axs[0, 1].spines['right'].set_visible(False)
# axs[0, 1].spines['top'].set_visible(False)
# axs[1, 0].hist(data_45min, 10,label='45min-30min')
# axs[1, 0].legend(loc='upper left',fontsize=8)
# axs[1, 0].spines['right'].set_visible(False)
# axs[1, 0].spines['top'].set_visible(False)
# axs[1, 1].hist(data_60min, 10,label='60min-45min')
# axs[1, 1].legend(loc='upper left',fontsize=8)
# axs[1, 1].spines['right'].set_visible(False)
# axs[1, 1].spines['top'].set_visible(False)
# pyplot.show()


# for k in range(len(data_15min)):
#     _, p_15min = stats.normaltest(np.array(data_15min[k]))
#     print(p_15min)
# for k in range(len(data_30min)):
#     _, p_30min = stats.normaltest(np.array(data_30min[k]))
#     print(p_30min)
# for k in range(len(data_45min)):
#     _, p_45min = stats.normaltest(np.array(data_45min[k]))
#     print(p_45min)
# for k in range(len(data_60min)):
#     _, p_60min = stats.normaltest(np.array(data_60min[k]))
#     print(p_60min)


# fig1, axs = pyplot.subplots(5,4,sharex=True,sharey=True)
# fig1.suptitle('Accuracy distribution using 45min-30min')
# axs[0, 0].hist(data_45min[0], 10,label='Fz',color='g')
# axs[0, 0].legend(loc='upper right',fontsize=8)
# axs[0, 0].spines['right'].set_visible(False)
# axs[0, 0].spines['top'].set_visible(False)
# axs[0, 1].hist(data_45min[1], 10,label='F3',color='g',alpha=0.5)
# axs[0, 1].legend(loc='upper right',fontsize=8)
# axs[0, 1].spines['right'].set_visible(False)
# axs[0, 1].spines['top'].set_visible(False)
# axs[0, 2].hist(data_45min[2], 10,label='F4',color='g')
# axs[0, 2].legend(loc='upper right',fontsize=8)
# axs[0, 2].spines['right'].set_visible(False)
# axs[0, 2].spines['top'].set_visible(False)
# axs[0, 3].hist(data_45min[3], 10,label='F7',color='g')
# axs[0, 3].legend(loc='upper right',fontsize=8)
# axs[0, 3].spines['right'].set_visible(False)
# axs[0, 3].spines['top'].set_visible(False)
# axs[1, 0].hist(data_45min[4], 10,label='F8',color='g')
# axs[1, 0].legend(loc='upper right',fontsize=8)
# axs[1, 0].spines['right'].set_visible(False)
# axs[1, 0].spines['top'].set_visible(False)
# axs[1, 1].hist(data_45min[5], 10,label='C4',color='g')
# axs[1, 1].legend(loc='upper right',fontsize=8)
# axs[1, 1].spines['right'].set_visible(False)
# axs[1, 1].spines['top'].set_visible(False)
# axs[1, 2].hist(data_45min[6], 10,label='C3',color='g')
# axs[1, 2].legend(loc='upper right',fontsize=8)
# axs[1, 2].spines['right'].set_visible(False)
# axs[1, 2].spines['top'].set_visible(False)
# axs[1, 3].hist(data_45min[7], 10,label='Cz',color='g')
# axs[1, 3].legend(loc='upper right',fontsize=8)
# axs[1, 3].spines['right'].set_visible(False)
# axs[1, 3].spines['top'].set_visible(False)
# axs[2, 0].hist(data_45min[8], 10,label='Pz',color='g')
# axs[2, 0].legend(loc='upper right',fontsize=8)
# axs[2, 0].spines['right'].set_visible(False)
# axs[2, 0].spines['top'].set_visible(False)
# axs[2, 1].hist(data_45min[9], 10,label='P3',color='g')
# axs[2, 1].legend(loc='upper right',fontsize=8)
# axs[2, 1].spines['right'].set_visible(False)
# axs[2, 1].spines['top'].set_visible(False)
# axs[2, 2].hist(data_45min[10], 10,label='P4',color='g')
# axs[2, 2].legend(loc='upper right',fontsize=8)
# axs[2, 2].spines['right'].set_visible(False)
# axs[2, 2].spines['top'].set_visible(False)
# axs[2, 3].hist(data_45min[11], 10,label='T3',color='g')
# axs[2, 3].legend(loc='upper right',fontsize=8)
# axs[2, 3].spines['right'].set_visible(False)
# axs[2, 3].spines['top'].set_visible(False)
# axs[3, 0].hist(data_45min[12], 10,label='T4',color='g')
# axs[3, 0].legend(loc='upper right',fontsize=8)
# axs[3, 0].spines['right'].set_visible(False)
# axs[3, 0].spines['top'].set_visible(False)
# axs[3, 1].hist(data_45min[13], 10,label='T5',color='g')
# axs[3, 1].legend(loc='upper right',fontsize=8)
# axs[3, 1].spines['right'].set_visible(False)
# axs[3, 1].spines['top'].set_visible(False)
# axs[3, 2].hist(data_45min[14], 10,label='T6',color='g')
# axs[3, 2].legend(loc='upper right',fontsize=8)
# axs[3, 2].spines['right'].set_visible(False)
# axs[3, 2].spines['top'].set_visible(False)
# axs[3, 3].hist(data_45min[15], 10,label='O1',color='g')
# axs[3, 3].legend(loc='upper right',fontsize=8)
# axs[3, 3].spines['right'].set_visible(False)
# axs[3, 3].spines['top'].set_visible(False)
# axs[4, 0].hist(data_45min[16], 10,label='O2',color='g')
# axs[4, 0].legend(loc='upper right',fontsize=8)
# axs[4, 0].spines['right'].set_visible(False)
# axs[4 ,0].spines['top'].set_visible(False)
# axs[4, 1].set_visible(False)
# axs[4, 2].set_visible(False)
# axs[4, 3].set_visible(False)
# pyplot.show()







# from scipy.stats import mannwhitneyu
#
# _,  res_30min = mannwhitneyu(data_15min, data_30min)
# _,  res_45min = mannwhitneyu(data_15min, data_45min)
# _,  res_60min = mannwhitneyu(data_15min, data_60min)
# print(res_30min);print(res_45min);print(res_60min);
#
# _,  res_45min = mannwhitneyu(data_30min, data_45min)
# _,  res_60min = mannwhitneyu(data_30min, data_60min)
# print(res_45min);print(res_60min);
#
# _,  res_60min = mannwhitneyu(data_45min, data_60min)
# print(res_60min);
#
#
# fig1, ax = pyplot.subplots()
# data=[data_15min,data_30min,data_45min,data_60min]
#
# bp1 = ax.boxplot(data, positions=[1,2,3,4], widths=0.4,
#                                      patch_artist=True, showfliers=False)
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
#         box.set(color='darkblue', linewidth=0.8)
#         box.set(facecolor='darkblue')
#
#
#
# locs, labels = pyplot.yticks([0.5,0.6,0.7],['0.5','0.6','0.7'],fontsize=12)
#
# locs, labels = pyplot.xticks([1,2,3,4],['15min-0min','30min-15min','45min-30min','60min-45min'],fontsize=10)
# # ax.hlines(0.5,0.5,24.5,'k',linestyles='--',alpha=0.5)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Preictal duration selection',fontsize=12)
# ax.set_ylabel('Accuracy across channels',fontsize=12)
# # ax.set_ylabel('AUC',fontsize=12)
# ax.set_title('Preictal epileptic seizures and preictal PNES classification',fontsize=12)
# # ax.legend([bp1["boxes"][0],bp1["boxes"][1], bp1["boxes"][2],bp1["boxes"][3]], ['15min','30min', '45min','60min'], loc='lower right',fontsize=7)
# pyplot.tight_layout()
# pyplot.show()
