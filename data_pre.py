from __future__ import division
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# # ### prepare for classifiers
##### prepare raw data
# ## Epileptic seizure raw
# # X_matrice=[]
# # directory =r'/Users/wxiong/Documents/PHD/2021.10/ML/ictal/ES/'
# # dir_list = list(os.scandir(directory))
# # dir_list.sort(key=lambda d:d.path)
# # for entry in dir_list:
# #     if (entry.path.endswith(".csv")) and entry.is_file():
# #         dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
# #         for m in range(int(len(dataset)/(256*60*2))):
# #         # for m in range(1):
# #             # for i in range(1):
# #             for i in range(0+256*60*2*m,256*60*1+256*60*2*m):
# #                 matrix = [dataset.loc[i, :]['Fz'],dataset.loc[i, :]['C4'],dataset.loc[i, :]['Pz'],dataset.loc[i, :]['C3'],
# #                           dataset.loc[i, :]['F3'],dataset.loc[i, :]['F4'],dataset.loc[i, :]['P4'],dataset.loc[i, :]['P3'],
# #                           dataset.loc[i, :]['T4'],dataset.loc[i, :]['T3'],dataset.loc[i, :]['O2'],dataset.loc[i, :]['O1'],
# #                           dataset.loc[i, :]['F7'],dataset.loc[i, :]['F8'],dataset.loc[i, :]['T6'],dataset.loc[i, :]['T5'],
# #                           dataset.loc[i, :]['Cz']]
# #                 X_matrice.append(matrix)
# #
# # channel=['Fz','C4','Pz', 'C3', 'F3', 'F4', 'P4', 'P3', 'T4', 'T3', 'O2', 'O1', 'F7', 'F8', 'T6', 'T5', 'Cz']
# # df = pd.DataFrame(X_matrice, columns=channel)
# # df.to_csv("/Users/wxiong/Documents/PHD/2021.10/ML/ictal/rawdata_ES_classifiers.csv")
# #
# # ## prepare for classifiers
# ##  PNES raw
# # X_matrice=[]
# # directory =r'/Users/wxiong/Documents/PHD/2021.10/ML/ictal/PNES/'
# # dir_list = list(os.scandir(directory))
# # dir_list.sort(key=lambda d:d.path)
# # for entry in dir_list:
# #     if (entry.path.endswith(".csv")) and entry.is_file():
# #         dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
# #         for m in range(int(len(dataset)/(256*60*2))):
# #         # for m in range(1):
# #             # for i in range(1):
# #             for i in range(0+256*60*2*m,256*60*1+256*60*2*m):
# #                 matrix = [dataset.loc[i, :]['Fz'],dataset.loc[i, :]['C4'],dataset.loc[i, :]['Pz'],dataset.loc[i, :]['C3'],
# #                           dataset.loc[i, :]['F3'],dataset.loc[i, :]['F4'],dataset.loc[i, :]['P4'],dataset.loc[i, :]['P3'],
# #                           dataset.loc[i, :]['T4'],dataset.loc[i, :]['T3'],dataset.loc[i, :]['O2'],dataset.loc[i, :]['O1'],
# #                           dataset.loc[i, :]['F7'],dataset.loc[i, :]['F8'],dataset.loc[i, :]['T6'],dataset.loc[i, :]['T5'],
# #                           dataset.loc[i, :]['Cz']]
# #                 X_matrice.append(matrix)
# #
# # channel=['Fz','C4','Pz', 'C3', 'F3', 'F4', 'P4', 'P3', 'T4', 'T3', 'O2', 'O1', 'F7', 'F8', 'T6', 'T5', 'Cz']
# # df = pd.DataFrame(X_matrice, columns=channel)
# # df.to_csv("/Users/wxiong/Documents/PHD/2021.10/ML/ictal/rawdata_PNES_classifiers.csv")
#
# #### combine ES and PNES raw for simple
# # dataset_1 = pd.read_csv("/Users/wxiong/Documents/PHD/2021.10/ML/ictal/rawdata_ES_ictal_classifiers.csv")
# # print(len(dataset_1))
# # dataset_1['type']=np.ones(len(dataset_1))
# #
# # dataset_2 = pd.read_csv("/Users/wxiong/Documents/PHD/2021.10/ML/ictal/rawdata_PNES_ictal_classifiers.csv")
# # print(len(dataset_2))
# # dataset_2['type']=np.zeros(len(dataset_2))
# #
# # data_sum=dataset_1.append(dataset_2, ignore_index=True)
# # data_sum.to_csv('C:/Users/wxiong/Documents/PHD/2021.10/ML/ictal/raw_PNES_and_ES_ictal.csv')
#
# # ## prepare for CNN
# # X_matrice=[]
# # Y_matrice=[]
# # directory =r'/Users/wxiong/Documents/PHD/2021.10/ML_preictal/15min/ES/'
# # dir_list = list(os.scandir(directory))
# # dir_list.sort(key=lambda d:d.path)
# # for entry in dir_list:
# #     if (entry.path.endswith(".csv")) and entry.is_file():
# #         dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
# #         # for m in range(int(len(dataset)/(256*15))):
# #         for m in range(1):
# #             # for i in range(0+256*15*m,256*15*1+256*15*m):
# #             # for i in range(1):
# #             i=256*15*m
# #             matrix = [
# #                 [dataset.loc[i, :]['F7'], dataset.loc[i, :]['F3'], dataset.loc[i, :]['Fz'], dataset.loc[i, :]['F4'],
# #                  dataset.loc[i, :]['F8']],
# #                 [dataset.loc[i, :]['T3'], dataset.loc[i, :]['C3'], dataset.loc[i, :]['Cz'], dataset.loc[i, :]['C4'],
# #                  dataset.loc[i, :]['T4']],
# #                 [dataset.loc[i, :]['T5'], dataset.loc[i, :]['P3'], dataset.loc[i, :]['Pz'], dataset.loc[i, :]['P4'],
# #                  dataset.loc[i, :]['T6']],
# #                 [0 * i, dataset.loc[i, :]['O1'], 0 * i, dataset.loc[i, :]['O2'], 0 * i]]
# #             X_matrice.append(matrix)
# #         Y_matrice.append(1)
# # # print(X_matrice)
# # # print(Y_matrice)
# # X_matrice=np.array(X_matrice)
# # # print(X_matrice.shape)
# # import itertools
# # data  = list(itertools.chain(*X_matrice))
# # df = pd.DataFrame.from_records(data)
# # df['label']=np.ones(len(df))
# # # print(df)
# # df.to_csv("/Users/wxiong/Documents/PHD/2021.10/ML_pre/data_ES_preictal_15min_CNN.csv")
# #
# # import os
# # X_matrice=[]
# # Y_matrice=[]
# # directory =r'/Users/wxiong/Documents/PHD/2021.10/ML_preictal/15min/PNES/'
# # dir_list = list(os.scandir(directory))
# # dir_list.sort(key=lambda d:d.path)
# # for entry in dir_list:
# #     if (entry.path.endswith(".csv")) and entry.is_file():
# #         dataset = pd.read_csv(entry.path, sep=',',skipinitialspace=True)
# #         # for m in range(int(len(dataset)/(256*15))):
# #         for m in range(1):
# #             # for i in range(0+256*15*m,256*60*1+256*15*m):
# #             # for i in range(1):
# #             i = 256 * 15 * m
# #             matrix = [
# #                 [dataset.loc[i, :]['F7'], dataset.loc[i, :]['F3'], dataset.loc[i, :]['Fz'], dataset.loc[i, :]['F4'],
# #                  dataset.loc[i, :]['F8']],
# #                 [dataset.loc[i, :]['T3'], dataset.loc[i, :]['C3'], dataset.loc[i, :]['Cz'], dataset.loc[i, :]['C4'],
# #                  dataset.loc[i, :]['T4']],
# #                 [dataset.loc[i, :]['T5'], dataset.loc[i, :]['P3'], dataset.loc[i, :]['Pz'], dataset.loc[i, :]['P4'],
# #                  dataset.loc[i, :]['T6']],
# #                 [0 * i, dataset.loc[i, :]['O1'], 0 * i, dataset.loc[i, :]['O2'], 0 * i]]
# #             X_matrice.append(matrix)
# #         Y_matrice.append(1)
# # # print(X_matrice)
# # # print(Y_matrice)
# # X_matrice=np.array(X_matrice)
# # # print(X_matrice.shape)
# #
# # import itertools
# # data  = list(itertools.chain(*X_matrice))
# # df = pd.DataFrame.from_records(data)
# # df['label']=np.zeros(len(df))
# # # print(df)
# # df.to_csv("/Users/wxiong/Documents/PHD/2021.10/ML_pre/data_PNES_preictal_15min_CNN.csv")




#


### combine two types of data
# import pandas as pd
# import os
# directory =r'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags/'
# keywords=['2']
# df_pnes = pd.DataFrame()
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         df = pd.read_csv(entry.path, skipinitialspace=True, sep=",")
#         df = df[df["tags"].isin(keywords)]
#         df_pnes=df_pnes.append(df, ignore_index=True)
# print(df_pnes);
# print(len(df_pnes));
# my_list=np.zeros(len(df_pnes))
# print(my_list);print(len(my_list))
# df_pnes['type']=my_list
# print(df_pnes);
# print(df_pnes.isnull().sum())
# directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags/'
# keywords=['2']
# df_es = pd.DataFrame()
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         df = pd.read_csv(entry.path, skipinitialspace=True, sep=",")
#         df = df[df["tags"].isin(keywords)]
#         df_es=df_es.append(df, ignore_index=True)
# print(df_es);
# print(len(df_es));
# my_list=np.ones(len(df_es))
# print(my_list);print(len(my_list))
# df_es['type']=my_list
# print(df_es);
# print(df_es.isnull().sum())
# data_sum=df_es.append(df_pnes, ignore_index=True)
# print(data_sum)
# data_sum.to_csv('C:/Users/wxiong/Documents/PHD/combine_features/PNES_and_ES_preictal.csv')


#
# ### combine data
# # import pandas as pd
# # import os
# # directory =r'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags/'
# # keywords=['0']
# # df_pnes = pd.DataFrame()
# # dir_list = list(os.scandir(directory))
# # dir_list.sort(key=lambda d:d.path)
# # for entry in dir_list:
# #     if (entry.path.endswith(".csv")) and entry.is_file():
# #         df = pd.read_csv(entry.path, skipinitialspace=True, sep=",")
# #         df = df[df["tags"].isin(keywords)]
# #         df_pnes=df_pnes.append(df, ignore_index=True)
# # print(df_pnes);
# # print(len(df_pnes));
# # my_list=np.zeros(len(df_pnes))
# # print(my_list);print(len(my_list))
# # df_pnes['type']=my_list
# # print(df_pnes);
# # print(df_pnes.isnull().sum())
# # directory =r'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags/'
# # keywords=['1']
# # df_es = pd.DataFrame()
# # dir_list = list(os.scandir(directory))
# # dir_list.sort(key=lambda d:d.path)
# # for entry in dir_list:
# #     if (entry.path.endswith(".csv")) and entry.is_file():
# #         df = pd.read_csv(entry.path, skipinitialspace=True, sep=",")
# #         df = df[df["tags"].isin(keywords)]
# #         df_es=df_es.append(df, ignore_index=True)
# # print(df_es);
# # print(len(df_es));
# # my_list=np.ones(len(df_es))
# # print(my_list);print(len(my_list))
# #

# # print(df_es);
# # print(df_es.isnull().sum())
# # data_sum=df_es.append(df_pnes, ignore_index=True)
# # print(data_sum)
# # data_sum.to_csv('C:/Users/wxiong/Documents/PHD/combine_features/PNES_interictal_ictal.csv')
#
#
# # ### combine two types of data
# # import pandas as pd
# # import os
# # directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags/'
# # keywords=['0']
# # df_pnes = pd.DataFrame()
# # dir_list = list(os.scandir(directory))
# # dir_list.sort(key=lambda d:d.path)
# # for entry in dir_list:
# #     if (entry.path.endswith(".csv")) and entry.is_file():
# #         df = pd.read_csv(entry.path, skipinitialspace=True, sep=",")
# #         df = df[df["tags"].isin(keywords)]
# #         df_pnes=df_pnes.append(df, ignore_index=True)
# # print(df_pnes);
# # print(len(df_pnes));
# # my_list=np.zeros(len(df_pnes))
# # print(my_list);print(len(my_list))
# # df_pnes['type']=my_list
# # print(df_pnes);
# # print(df_pnes.isnull().sum())
# # directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags/'
# # keywords=['1']
# # df_es = pd.DataFrame()
# # dir_list = list(os.scandir(directory))
# # dir_list.sort(key=lambda d:d.path)
# # for entry in dir_list:
# #     if (entry.path.endswith(".csv")) and entry.is_file():
# #         df = pd.read_csv(entry.path, skipinitialspace=True, sep=",")
# #         df = df[df["tags"].isin(keywords)]
# #         df_es=df_es.append(df, ignore_index=True)
# # print(df_es);
# # print(len(df_es));
# # my_list=np.ones(len(df_es))
# # print(my_list);print(len(my_list))
# # df_es['type']=my_list
# # print(df_es);
# # print(df_es.isnull().sum())
# # data_sum=df_es.append(df_pnes, ignore_index=True)
# # print(data_sum)
# # data_sum.to_csv('C:/Users/wxiong/Documents/PHD/combine_features/ES_interictal_ictal.csv')
#
#



# ## critical slowing feature combination
# directory =r'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags_15min/test'
# df_pnes = pd.DataFrame()
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         df = pd.read_csv(entry.path, skipinitialspace=True, sep=",")
#         indice = df.index[df["tags"] == 2].tolist()
#         # indice_new_test =sorted(indice)
#         # indice_new=  indice_new_test [40:60]
#         # df_pnes = df_pnes.append(df.iloc[indice_new], ignore_index=True)
#         # index_new_0 = [x - 40 for x in indice]
#         # df_pnes = df_pnes.append(df.iloc[index_new_0], ignore_index=True)
#         # index_new_0 = [x - 60 for x in indice]
#         # df_pnes = df_pnes.append(df.iloc[index_new_0], ignore_index=True)
#         index_new_1 = [x - 120 for x in indice]
#         df_pnes = df_pnes.append(df.iloc[index_new_1], ignore_index=True)
#         # index_new_2 = [x - 180 for x in indice]
#         # df_pnes = df_pnes.append(df.iloc[index_new_2], ignore_index=True)
#         # index_new_3 = [x - 240 for x in indice]
#         # df_pnes = df_pnes.append(df.iloc[index_new_3], ignore_index=True)
#         # index_new_4 = [x - 300 for x in indice]
#         # df_pnes = df_pnes.append(df.iloc[index_new_4], ignore_index=True)
#         # index_new_5 = [x - 360 for x in indice]
#         # df_pnes = df_pnes.append(df.iloc[index_new_5], ignore_index=True)
#         # index_new_6 = [x - 420 for x in indice]
#         # df_pnes = df_pnes.append(df.iloc[index_new_6], ignore_index=True)
#
#         # ind = df.index[df["tags"] == 0].tolist()
#         # ind_modify_0 = [elem for elem in ind if elem not in index_new_0]
#         # ind_modify_1 =[elem for elem in ind_modify_0 if elem not in index_new_1]
#         # ind_modify = [elem for elem in ind_modify_1 if elem not in index_new_2]
#         # df_pnes = df_pnes.append(df.iloc[ind_modify], ignore_index=True)
#
# # directory =r'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags/'
# # keywords=['1']
# # df_pnes = pd.DataFrame()
# # dir_list.sort(key=lambda d:d.path)
# # for entry in dir_list:
# #     if (entry.path.endswith(".csv"
# # dir_list = list(os.scandir(directory)))) and entry.is_file():
# #         df = pd.read_csv(entry.path, skipinitialspace=True, sep=",")
# #         df = df[df["tags"].isin(keywords)]
# #         df_pnes=df_pnes.append(df, ignore_index=True)
#
# print(df_pnes);
# print(len(df_pnes));
# my_list=np.zeros(len(df_pnes))
# print(my_list);print(len(my_list))
# df_pnes['type']=my_list
# print(df_pnes);
# print(df_pnes.isnull().sum())
#
#
# directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_15min/test'
# df_es = pd.DataFrame()
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         df = pd.read_csv(entry.path, skipinitialspace=True, sep=",")
#         indice = df.index[df["tags"] == 2].tolist()
#         # indice_new_test = sorted(indice)
#         # indice_new = indice_new_test[40:60]
#         # df_es = df_es.append(df.iloc[indice], ignore_index=True)
#         index_new = [x - 120 for x in indice]
#         df_es=df_es.append(df.iloc[index_new], ignore_index=True)
#
# print(df_es);
# print(len(df_es));
# my_list=2*np.ones(len(df_es))
# # my_list=np.ones(len(df_es))
# print(my_list);print(len(my_list))
# df_es['type']=my_list
# print(df_es);
# print(df_es.isnull().sum())
# data_sum=df_es.append(df_pnes, ignore_index=True)
# print(data_sum)
# data_sum.to_csv('C:/Users/wxiong/Documents/PHD/combine_features/PNES_preictal_and_ES_preictal(60min).csv')





# # critical slowing feature 30min pre-ictal tags
# directory =r'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags_105_90_min'
# df_pnes = pd.DataFrame()
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         df = pd.read_csv(entry.path, skipinitialspace=True, sep=",")
#         Patient_ID=entry.name.split('_')[0]
#         # print(len(df));
#         print(len(df[df["tags"] == 8]));
#         indice = df.index[df["tags"] == 8].tolist()
#         index_new_0 = [x - 60 for x in indice if x >= 60]
#         # index_new_1 = [x - 60 for x in index_new_0 if x >= 60]
#         # index_new_2 = [x - 60 for x in index_new_1 if x >= 60]
#         df.loc[index_new_0,"tags"] = 9
#         # df.loc[index_new_1, "tags"] = 2
#         # df.loc[index_new_2, "tags"] = 2
#         print(len(df[df["tags"] == 9]));
#         # df.to_csv(f'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags_30_15_min/{Patient_ID}_and_tags.csv') # 3 tags
#         # df.to_csv(f'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags_45_30_min/{Patient_ID}_and_tags.csv') # 4 tags
#         # df.to_csv(f'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags_60_45_min/{Patient_ID}_and_tags.csv') # 5 tags
#         df.to_csv(
#             f'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags_120_105_min/{Patient_ID}_and_tags.csv')  # 6 tags





# ### cycles of critical slowing feature
# channels = ['Fz_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'F7_EEGvar', 'F8_EEGvar', 'C4_EEGvar', 'C3_EEGvar', 'Cz_EEGvar',
#             'Pz_EEGvar', 'P3_EEGvar', 'P4_EEGvar', 'T3_EEGvar', 'T4_EEGvar', 'T5_EEGvar', 'T6_EEGvar', 'O1_EEGvar',
#             'O2_EEGvar',
#
#             'Fz_EEGauto', 'F3_EEGauto', 'F4_EEGauto', 'F7_EEGauto', 'F8_EEGauto', 'C4_EEGauto', 'C3_EEGauto','Cz_EEGauto',
#             'Pz_EEGauto', 'P3_EEGauto', 'P4_EEGauto', 'T3_EEGauto', 'T4_EEGauto', 'T5_EEGauto', 'T6_EEGauto','O1_EEGauto',
#             'O2_EEGauto',
#             'ch31_RRIvar','ch31_RRIauto']
#
# directory =r'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags_60_45_min'
# df_pnes = pd.DataFrame()
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         df = pd.read_csv(entry.path, skipinitialspace=True, sep=",")
#         Patient_ID=entry.name.split('_')[0]
#         for ch in channels:
#             df['6h' + ch] = df[ch].rolling(1440).mean()
#             df['12h' + ch] = df[ch].rolling(1440 * 2).mean()
#             df['24h' + ch] = df[ch].rolling(1440 * 4).mean()
#         df.to_csv(f'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags_60_45_min/cycles/{Patient_ID}_and_tags.csv')


## critical slowing feature combination
directory =r'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags_120_105_min/'
df_pnes = pd.DataFrame()
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        df = pd.read_csv(entry.path, skipinitialspace=True, sep=",")
        indice = df.index[df["tags"] == 9].tolist()
        indice_new =sorted(indice)
        df_pnes = df_pnes.append(df.iloc[indice_new], ignore_index=True)

print(df_pnes);
print(len(df_pnes));
my_list=np.zeros(len(df_pnes))
print(my_list);print(len(my_list))
df_pnes['type']=my_list
print(df_pnes);
print(df_pnes.isnull().sum())


directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_120_105_min/'
df_es = pd.DataFrame()
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d:d.path)
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        df = pd.read_csv(entry.path, skipinitialspace=True, sep=",")
        indice = df.index[df["tags"] == 9].tolist()
        indice_new = sorted(indice)
        df_es = df_es.append(df.iloc[indice_new], ignore_index=True)


print(df_es);
print(len(df_es));
my_list=np.ones(len(df_es))
# my_list=np.ones(len(df_es))
print(my_list);print(len(my_list))
df_es['type']=my_list
print(df_es);
print(df_es.isnull().sum())
data_sum=df_es.append(df_pnes, ignore_index=True)
print(data_sum)
data_sum.to_csv('C:/Users/wxiong/Documents/PHD/combine_features/PNES_preictal_and_ES_preictal(120_105min)_raw.csv')




# df_es = pd.DataFrame()
# feature_list_sum=[]
# tag_list_sum=[]
# directory =r'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags_30min/test'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         df = pd.read_csv(entry.path, skipinitialspace=True, sep=",")
#         Patient_ID=entry.name.split('_')[0]
#
#         feature_list_inter = []
#         tag_list_inter = []
#         df_inter = df[df["tags"] == 0]
#         df_inter_splits = np.array_split(df_inter.iloc[:,2:40], int(len(df_inter)/40))
#         for i in range(len(df_inter_splits)):
#             feature_list_inter.append(df_inter_splits[i].values.tolist())
#             tag_list_inter.append(0)
#         # print(len(feature_list));
#
#         feature_list_pre = []
#         tag_list_pre = []
#         df_pre = df[df["tags"] == 2]
#         # df_pre['index_old']=df_inter.iloc[:,1]
#         df_pre_splits = np.array_split(df_pre.iloc[:, 2:40], int(len(df_pre) / 40))
#         for i in range(len(df_pre_splits)):
#             feature_list_pre.append(df_pre_splits[i].values.tolist())
#             tag_list_pre.append(2)
#         # print(len(feature_list));
#
#     feature_list_sum = feature_list_sum + feature_list_inter + feature_list_pre
#     tag_list_sum = tag_list_sum + tag_list_inter + tag_list_pre
#
#
# df_es['features'] = feature_list_sum
# df_es['tags'] = tag_list_sum
# print(df_es)
# df_es.to_csv(f'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags_30min/PNES_test.csv')





# import pandas as pd
# import os
# # directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/test/'
# directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_45min/test'
# keywords=[0]
# df_pnes = pd.DataFrame()
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         df = pd.read_csv(entry.path, skipinitialspace=True, sep=",")
#
#         df = df[df["tags"].isin(keywords)]
#         print(df)
#         # df = df.groupby(np.arange(len(df)) // 20).mean()
#         df = df.groupby(np.arange(len(df)) // 4).mean()
#         df_pnes=df_pnes.append(df, ignore_index=True)
# print(df_pnes);
# print(len(df_pnes));
# my_list=np.zeros(len(df_pnes))
# print(my_list);print(len(my_list))
# df_pnes['tags']=my_list
# print(df_pnes);
# print(df_pnes.isnull().sum())
# # directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_30min/test/'
# directory =r'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_45min/test'
# keywords=[2]
# df_es = pd.DataFrame()
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d:d.path)
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         df = pd.read_csv(entry.path, skipinitialspace=True, sep=",")
#         df = df[df["tags"].isin(keywords)]
#         # df = df.groupby(np.arange(len(df)) // 20).mean()
#         df = df.groupby(np.arange(len(df)) // 4).mean()
#         df_es=df_es.append(df, ignore_index=True)
# print(df_es);
# print(len(df_es));
# my_list=2*np.ones(len(df_es))
# print(my_list);print(len(my_list))
# df_es['tags']=my_list
# print(df_es);
# print(df_es.isnull().sum())
# data_sum=df_es.append(df_pnes, ignore_index=True)
# print(data_sum)
# data_sum.to_csv('C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_45min/ES_interictal_preictal_average_1min.csv')





# ## combine csv
# import os, glob
# import pandas as pd
#
# path = "C:/Users/wxiong/Documents/PHD/2021.10/ML_preictal/PNES_preictal"
#
# all_files = glob.glob(os.path.join(path, "*.csv"))
# df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
# df_merged   = pd.concat(df_from_each_file, ignore_index=True)
# df_merged.to_csv( "C:/Users/wxiong/Documents/PHD/2021.10/ML_preictal/PNES_preictal/PNES_merged.csv")










