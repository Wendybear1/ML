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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from scipy.signal import lfilter
from sklearn.preprocessing import StandardScaler,LabelEncoder
import tensorflow as tf




# ## classfication preictal and preictal data using cs features
# ## try 15min, 30min, 45 min and 60min compare
# channels = ['Fz_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'F7_EEGvar', 'F8_EEGvar', 'C4_EEGvar', 'C3_EEGvar', 'Cz_EEGvar',
#             'Pz_EEGvar', 'P3_EEGvar', 'P4_EEGvar', 'T3_EEGvar', 'T4_EEGvar', 'T5_EEGvar', 'T6_EEGvar', 'O1_EEGvar',
#             'O2_EEGvar',
#
#             'Fz_EEGauto', 'F3_EEGauto', 'F4_EEGauto', 'F7_EEGauto', 'F8_EEGauto', 'C4_EEGauto', 'C3_EEGauto','Cz_EEGauto',
#             'Pz_EEGauto', 'P3_EEGauto', 'P4_EEGauto', 'T3_EEGauto', 'T4_EEGauto', 'T5_EEGauto', 'T6_EEGauto','O1_EEGauto',
#             'O2_EEGauto']
#
# score_KNN_sum = []; recall_KNN = []; F1_KNN = [];roc_auc_KNN_sum=[]
# score_DT_sum = []; recall_DT = []; F1_DT = [];roc_auc_DT_sum=[]
# score_RFT_sum = []; recall_RFT = []; F1_RFT = [];roc_auc_RFT_sum=[]
# score_NB_sum = []; recall_NB = []; F1_NB = [];roc_auc_NB_sum=[]
# score_SVM_sum = []; recall_SVM = []; F1_SVM = []; roc_auc_SVM_sum=[]
#
# for m in range(1):
#     score_KNN = [];score_DT = [];score_RFT = [];score_NB = [];score_SVM = [];
#     roc_auc_KNN=[];roc_auc_DT = [];roc_auc_RFT = [];roc_auc_NB = [];roc_auc_SVM = [];
#     for i in range(1):
#     # for i in range(100):
#         ### evaluate algorithms
#         dataset = pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/PNES_preictal_and_ES_preictal(15_0min)_raw.csv',sep=',')
#         column_name = dataset.columns[6:].tolist()
#
#         class_count_0, class_count_1 = dataset['type'].value_counts()
#         print(class_count_0); print(class_count_1)
#         class_0 = dataset[dataset['type'] == 1]
#         class_1 = dataset[dataset['type'] == 0]
#         print('class 0:', class_0.shape);
#         print('class 1:', class_1.shape);
#         class_0_under = class_0.sample(class_count_1)
#         test_under = pd.concat([class_0_under, class_1], axis=0)
#
#
#         X = test_under[['Fz_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'F7_EEGvar', 'F8_EEGvar', 'C4_EEGvar', 'C3_EEGvar', 'Cz_EEGvar',
#             'Pz_EEGvar', 'P3_EEGvar', 'P4_EEGvar', 'T3_EEGvar', 'T4_EEGvar', 'T5_EEGvar', 'T6_EEGvar', 'O1_EEGvar',
#             'O2_EEGvar',
#
#             'Fz_EEGauto', 'F3_EEGauto', 'F4_EEGauto', 'F7_EEGauto', 'F8_EEGauto', 'C4_EEGauto', 'C3_EEGauto','Cz_EEGauto',
#             'Pz_EEGauto', 'P3_EEGauto', 'P4_EEGauto', 'T3_EEGauto', 'T4_EEGauto', 'T5_EEGauto', 'T6_EEGauto','O1_EEGauto',
#             'O2_EEGauto'
#            ]]
#         # X = test_under[[channels[m], channels[m + 17]]]
#         # X = test_under[[channels[m],channels[m+17],'ch31_RRIvar', 'ch31_RRIauto']]
#         # X = test_under[['ch31_RRIvar', 'ch31_RRIauto']]
#
#         y = test_under[['type']]
#         y = y.copy()
#         # y['type']=pd.get_dummies(y['type'])
#
#
#         X_train, X_validation, Y_train, Y_validation = train_test_split(X, y.values.ravel(), test_size=0.20, random_state=1)
#         X_train_1 = tf.keras.utils.normalize(X_train, axis=1)
#         X_validation_1 = tf.keras.utils.normalize(X_validation, axis=1)
#
#         # # Make predictions on validation dataset
#         model = KNeighborsClassifier()
#         model.fit(X_train_1, Y_train)
#         predictions = model.predict(X_validation_1)
#         score_KNN.append(accuracy_score(Y_validation, predictions))
#         # recall_KNN.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_KNN.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_KNN.append(roc_auc_score(Y_validation, predictions))
#
#         model = DecisionTreeClassifier()
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_DT.append(accuracy_score(Y_validation, predictions))
#         # recall_DT.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_DT.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_DT.append(roc_auc_score(Y_validation, predictions))
#
#         model = RandomForestClassifier(n_estimators=100)
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_RFT.append(accuracy_score(Y_validation, predictions))
#         # recall_RFT.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_RFT.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_RFT.append(roc_auc_score(Y_validation, predictions))
#
#         model = GaussianNB()
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_NB.append(accuracy_score(Y_validation, predictions))
#         # recall_NB.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_NB.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_NB.append(roc_auc_score(Y_validation, predictions))
#
#
#         model = SVC(gamma='auto')
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_SVM.append(accuracy_score(Y_validation, predictions))
#         # recall_SVM.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_SVM.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_SVM.append(roc_auc_score(Y_validation, predictions))
#
#     score_KNN_sum.append(score_KNN);score_DT_sum.append(score_DT);score_RFT_sum.append(score_RFT);
#     score_NB_sum.append(score_NB);score_SVM_sum.append(score_SVM);
#     roc_auc_KNN_sum.append(roc_auc_KNN);roc_auc_DT_sum.append(roc_auc_DT);roc_auc_RFT_sum.append(roc_auc_RFT);
#     roc_auc_NB_sum.append(roc_auc_NB);roc_auc_SVM_sum.append(roc_auc_SVM);
#
# print(np.mean(score_KNN_sum));print(np.mean(score_DT_sum));print(np.mean(score_RFT_sum));print(np.mean(score_NB_sum));
# print(np.mean(score_SVM_sum));
# for n in range(len(score_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=score_KNN_sum[n]
#     df['DT']=score_DT_sum[n]
#     df['RFT']=score_RFT_sum[n]
#     df['NB']=score_NB_sum[n]
#     df['SVM']=score_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\60min_ALLchannels_ECGEperformance_Accuracy_check.csv')
#
#
# for n in range(len(roc_auc_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=roc_auc_KNN_sum[n]
#     df['DT']=roc_auc_DT_sum[n]
#     df['RFT']=roc_auc_RFT_sum[n]
#     df['NB']=roc_auc_NB_sum[n]
#     df['SVM']=roc_auc_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\60min_ALLchannels_ECGperformance_AUC_check.csv')






channels = ['Fz_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'F7_EEGvar', 'F8_EEGvar', 'C4_EEGvar', 'C3_EEGvar', 'Cz_EEGvar',
            'Pz_EEGvar', 'P3_EEGvar', 'P4_EEGvar', 'T3_EEGvar', 'T4_EEGvar', 'T5_EEGvar', 'T6_EEGvar', 'O1_EEGvar',
            'O2_EEGvar',

            'Fz_EEGauto', 'F3_EEGauto', 'F4_EEGauto', 'F7_EEGauto', 'F8_EEGauto', 'C4_EEGauto', 'C3_EEGauto','Cz_EEGauto',
            'Pz_EEGauto', 'P3_EEGauto', 'P4_EEGauto', 'T3_EEGauto', 'T4_EEGauto', 'T5_EEGauto', 'T6_EEGauto','O1_EEGauto',
            'O2_EEGauto',

            'ch31_RRIvar', 'ch31_RRIauto',

            'type',]

# channels = ['6hFz_EEGvar', '6hF3_EEGvar', '6hF4_EEGvar', '6hF7_EEGvar', '6hF8_EEGvar', '6hC4_EEGvar', '6hC3_EEGvar',
#             '6hCz_EEGvar',
#             '6hPz_EEGvar', '6hP3_EEGvar', '6hP4_EEGvar', '6hT3_EEGvar', '6hT4_EEGvar', '6hT5_EEGvar', '6hT6_EEGvar',
#             '6hO1_EEGvar',
#             '6hO2_EEGvar',
#
#             '6hFz_EEGauto', '6hF3_EEGauto', '6hF4_EEGauto', '6hF7_EEGauto', '6hF8_EEGauto', '6hC4_EEGauto',
#             '6hC3_EEGauto', '6hCz_EEGauto',
#             '6hPz_EEGauto', '6hP3_EEGauto', '6hP4_EEGauto', '6hT3_EEGauto', '6hT4_EEGauto', '6hT5_EEGauto',
#             '6hT6_EEGauto', '6hO1_EEGauto',
#             '6hO2_EEGauto',
#
#             'type',
#             ]

# channels = [
#             'Fz_EEGauto',  'F4_EEGauto', 'F7_EEGauto', 'C4_EEGauto',
#             'C3_EEGauto', 'Cz_EEGauto',
#             'Pz_EEGauto', 'P3_EEGauto',  'T3_EEGauto', 'T5_EEGauto',
#             'type',
#             ]


score_KNN_sum = []; recall_KNN_sum = []; F1_KNN_sum = [];roc_auc_KNN_sum=[]
score_DT_sum = []; recall_DT_sum = []; F1_DT_sum = [];roc_auc_DT_sum=[]
score_RFT_sum = []; recall_RFT_sum = []; F1_RFT_sum = [];roc_auc_RFT_sum=[]
score_NB_sum = []; recall_NB_sum = []; F1_NB_sum = [];roc_auc_NB_sum=[]
score_SVM_sum = []; recall_SVM_sum = []; F1_SVM_sum = []; roc_auc_SVM_sum=[]
for m in range(1):
    score_KNN = [];score_DT = [];score_RFT = [];score_NB = [];score_SVM = [];
    F1_KNN = [];F1_DT = [];F1_RFT = [];F1_NB = [];F1_SVM = [];
    recall_KNN=[];recall_DT = [];recall_RFT = [];recall_NB = [];recall_SVM = [];
    roc_auc_KNN = [];roc_auc_DT = [];roc_auc_RFT = [];roc_auc_NB = [];roc_auc_SVM = [];
    for i in range(100):
        ### evaluate algorithms
        dataset = pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/PNES_preictal_and_ES_preictal(60_45min)_raw.csv',sep=',')
        column_name = dataset.columns[6:].tolist()
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

        X = test_under[channels[0:-3]]
        # X = test_under[channels[0:-1]]
        # X = test_under[channels[-3:-1]]

        y = test_under[channels[-1]]


        X_train, X_validation, Y_train, Y_validation = train_test_split(X, y.values.ravel(), test_size=0.20, random_state=1)
        X_train_1 = tf.keras.utils.normalize(X_train, axis=1)
        X_validation_1 = tf.keras.utils.normalize(X_validation, axis=1)

        # # Make predictions on validation dataset
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_1, Y_train)
        predictions = model.predict(X_validation_1)
        score_KNN.append(accuracy_score(Y_validation, predictions))
        recall_KNN.append(recall_score(Y_validation, predictions, average='weighted'))
        F1_KNN.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_KNN.append(roc_auc_score(Y_validation, predictions))

        model = DecisionTreeClassifier()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_DT.append(accuracy_score(Y_validation, predictions))
        recall_DT.append(recall_score(Y_validation, predictions, average='weighted'))
        F1_DT.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_DT.append(roc_auc_score(Y_validation, predictions))

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_RFT.append(accuracy_score(Y_validation, predictions))
        recall_RFT.append(recall_score(Y_validation, predictions, average='weighted'))
        F1_RFT.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_RFT.append(roc_auc_score(Y_validation, predictions))

        model = GaussianNB()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_NB.append(accuracy_score(Y_validation, predictions))
        recall_NB.append(recall_score(Y_validation, predictions, average='weighted'))
        F1_NB.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_NB.append(roc_auc_score(Y_validation, predictions))


        model = SVC(gamma='auto', kernel='rbf')
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_SVM.append(accuracy_score(Y_validation, predictions))
        recall_SVM.append(recall_score(Y_validation, predictions, average='weighted'))
        F1_SVM.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_SVM.append(roc_auc_score(Y_validation, predictions))

    score_KNN_sum.append(score_KNN);score_DT_sum.append(score_DT);score_RFT_sum.append(score_RFT);
    score_NB_sum.append(score_NB);score_SVM_sum.append(score_SVM);
    roc_auc_KNN_sum.append(roc_auc_KNN);roc_auc_DT_sum.append(roc_auc_DT);roc_auc_RFT_sum.append(roc_auc_RFT);
    roc_auc_NB_sum.append(roc_auc_NB);roc_auc_SVM_sum.append(roc_auc_SVM);
    F1_KNN_sum.append(F1_KNN);F1_DT_sum.append(F1_DT);F1_RFT_sum.append(F1_RFT);
    F1_NB_sum.append(F1_NB);F1_SVM_sum.append(F1_SVM);
    recall_KNN_sum.append(recall_KNN);recall_DT_sum.append(recall_DT);recall_RFT_sum.append(recall_RFT);
    recall_NB_sum.append(recall_NB);recall_SVM_sum.append(recall_SVM);

print(np.median(score_KNN_sum));print(np.median(score_DT_sum));print(np.median(score_RFT_sum));print(np.median(score_NB_sum));
print(np.median(score_SVM_sum));

# for n in range(len(score_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=score_KNN_sum[n]
#     df['DT']=score_DT_sum[n]
#     df['RFT']=score_RFT_sum[n]
#     df['NB']=score_NB_sum[n]
#     df['SVM']=score_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\allchannels_June\\raw_15_0min_channels_EEGECGperformance_Accuracy_KNN10_SVMrbf_RF200.csv')


# for n in range(len(roc_auc_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=roc_auc_KNN_sum[n]
#     df['DT']=roc_auc_DT_sum[n]
#     df['RFT']=roc_auc_RFT_sum[n]
#     df['NB']=roc_auc_NB_sum[n]
#     df['SVM']=roc_auc_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\allchannels_June\\raw_120_105min_channels_EEGperformance_AUC.csv')

for n in range(len(F1_KNN_sum)):
    df = pd.DataFrame()
    df['KNN']=F1_KNN_sum[n]
    df['DT']=F1_DT_sum[n]
    df['RFT']=F1_RFT_sum[n]
    df['NB']=F1_NB_sum[n]
    df['SVM']=F1_SVM_sum[n]
    df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\allchannels_June\\raw_60_45min_channels_EEGperformance_F1_x.csv')

for n in range(len(recall_KNN_sum)):
    df = pd.DataFrame()
    df['KNN']=recall_KNN_sum[n]
    df['DT']=recall_DT_sum[n]
    df['RFT']=recall_RFT_sum[n]
    df['NB']=recall_NB_sum[n]
    df['SVM']=recall_SVM_sum[n]
    df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\allchannels_June\\raw_60_45min_channels_EEGperformance_recall_x.csv')






# channels = ['Fz_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'F7_EEGvar', 'F8_EEGvar', 'C4_EEGvar', 'C3_EEGvar', 'Cz_EEGvar',
#             'Pz_EEGvar', 'P3_EEGvar', 'P4_EEGvar', 'T3_EEGvar', 'T4_EEGvar', 'T5_EEGvar', 'T6_EEGvar', 'O1_EEGvar',
#             'O2_EEGvar',
#
#             'Fz_EEGauto', 'F3_EEGauto', 'F4_EEGauto', 'F7_EEGauto', 'F8_EEGauto', 'C4_EEGauto', 'C3_EEGauto','Cz_EEGauto',
#             'Pz_EEGauto', 'P3_EEGauto', 'P4_EEGauto', 'T3_EEGauto', 'T4_EEGauto', 'T5_EEGauto', 'T6_EEGauto','O1_EEGauto',
#             'O2_EEGauto',
#
#             'ch31_RRIvar', 'ch31_RRIauto',
#
#             'type',]
#
# ### staitistic test
# import random
# score_KNN_sum = []; recall_KNN = []; F1_KNN = [];roc_auc_KNN_sum=[]
# score_DT_sum = []; recall_DT = []; F1_DT = [];roc_auc_DT_sum=[]
# score_RFT_sum = []; recall_RFT = []; F1_RFT = [];roc_auc_RFT_sum=[]
# score_NB_sum = []; recall_NB = []; F1_NB = [];roc_auc_NB_sum=[]
# score_SVM_sum = []; recall_SVM = []; F1_SVM = []; roc_auc_SVM_sum=[]
#
# # for m in range(1):
# #     score_KNN = [];score_DT = [];score_RFT = [];score_NB = [];score_SVM = [];
# #     roc_auc_KNN=[];roc_auc_DT = [];roc_auc_RFT = [];roc_auc_NB = [];roc_auc_SVM = [];
# #     for i in range(1000):
# #         ### evaluate algorithms
# #         dataset = pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/PNES_preictal_and_ES_preictal(15_0min)_raw.csv',sep=',')
# #         column_name = dataset.columns[6:].tolist()
# #         print(column_name)
# #         df=dataset[channels]
# #         df=df.dropna()
# #         class_count_0, class_count_1 = df['type'].value_counts()
# #         print(class_count_0); print(class_count_1)
# #         class_0 = df[df['type'] == 1]
# #         class_1 = df[df['type'] == 0]
# #         print('class 0:', class_0.shape);
# #         print('class 1:', class_1.shape);
# #         tag_list=[0]*class_count_1+[1]*class_count_0
# #         random.shuffle(tag_list)
# #         df['type']=tag_list
# #         class_0_under = class_0.sample(class_count_1)
# #         test_under = pd.concat([class_0_under, class_1], axis=0)
# #         X = test_under[channels[0:-3]]
# #         # X = test_under[channels[0:-1]]
# #         # X = test_under[channels[-3:-1]]
# #         y = test_under[channels[-1]]
# #         X_train, X_validation, Y_train, Y_validation = train_test_split(X, y.values.ravel(), test_size=0.20, random_state=1)
# #         X_train_1 = tf.keras.utils.normalize(X_train, axis=1)
# #         X_validation_1 = tf.keras.utils.normalize(X_validation, axis=1)
#
# for m in range(1):
#     score_KNN = [];score_DT = [];score_RFT = [];score_NB = [];score_SVM = [];
#     F1_KNN = [];F1_DT = [];F1_RFT = [];F1_NB = [];F1_SVM = [];
#     recall_KNN=[];recall_DT = [];recall_RFT = [];recall_NB = [];recall_SVM = [];
#     roc_auc_KNN = [];roc_auc_DT = [];roc_auc_RFT = [];roc_auc_NB = [];roc_auc_SVM = [];
#     for i in range(100):
#         ### evaluate algorithms
#         dataset = pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/PNES_preictal_and_ES_preictal(60_45min)_raw.csv',sep=',')
#         column_name = dataset.columns[6:].tolist()
#         print(column_name)
#
#         df=dataset[channels]
#         df=df.dropna()
#
#         class_count_0, class_count_1 = df['type'].value_counts()
#         print(class_count_0); print(class_count_1)
#         class_0 = df[df['type'] == 1]
#         class_1 = df[df['type'] == 0]
#         print('class 0:', class_0.shape);
#         print('class 1:', class_1.shape);
#         class_0_under = class_0.sample(class_count_1)
#         test_under = pd.concat([class_0_under, class_1], axis=0)
#
#         # X = test_under[channels[0:-3]]
#         # X = test_under[channels[0:-1]]
#         X = test_under[channels[-3:-1]]
#         y = test_under[channels[-1]]
#
#
#         X_train, X_validation, Y_train, Y_validation = train_test_split(X, y.values.ravel(), test_size=0.20, random_state=1)
#         X_train_1 = tf.keras.utils.normalize(X_train, axis=1)
#         X_validation_1 = tf.keras.utils.normalize(X_validation, axis=1)
#
#
#         label_0=np.unique(Y_validation, return_counts=True)[0][0]
#         label_1=np.unique(Y_validation, return_counts=True)[0][1]
#         class_count_0 = np.unique(Y_validation, return_counts=True)[1][0]
#         class_count_1 = np.unique(Y_validation, return_counts=True)[1][1]
#
#         print(class_count_0); print(class_count_1)
#         tag_list=[label_0]*class_count_1+[label_1]*class_count_0
#         random.shuffle(tag_list)
#         Y_validation=tag_list
#
#         # # Make predictions on validation dataset
#         model = KNeighborsClassifier()
#         model.fit(X_train_1, Y_train)
#         predictions = model.predict(X_validation_1)
#         score_KNN.append(accuracy_score(Y_validation, predictions))
#         # recall_KNN.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_KNN.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_KNN.append(roc_auc_score(Y_validation, predictions))
#
#         model = DecisionTreeClassifier()
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_DT.append(accuracy_score(Y_validation, predictions))
#         # recall_DT.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_DT.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_DT.append(roc_auc_score(Y_validation, predictions))
#
#         model = RandomForestClassifier(n_estimators=100)
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_RFT.append(accuracy_score(Y_validation, predictions))
#         # recall_RFT.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_RFT.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_RFT.append(roc_auc_score(Y_validation, predictions))
#
#         model = GaussianNB()
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_NB.append(accuracy_score(Y_validation, predictions))
#         # recall_NB.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_NB.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_NB.append(roc_auc_score(Y_validation, predictions))
#
#
#         model = SVC(gamma='auto')
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_SVM.append(accuracy_score(Y_validation, predictions))
#         # recall_SVM.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_SVM.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_SVM.append(roc_auc_score(Y_validation, predictions))
#
#         score_KNN_sum.append(score_KNN);score_DT_sum.append(score_DT);score_RFT_sum.append(score_RFT);
#         score_NB_sum.append(score_NB);score_SVM_sum.append(score_SVM);
#         roc_auc_KNN_sum.append(roc_auc_KNN);roc_auc_DT_sum.append(roc_auc_DT);roc_auc_RFT_sum.append(roc_auc_RFT);
#         roc_auc_NB_sum.append(roc_auc_NB);roc_auc_SVM_sum.append(roc_auc_SVM);
#
#
#     roc_auc_KNN_sum.append(roc_auc_KNN);roc_auc_DT_sum.append(roc_auc_DT);roc_auc_RFT_sum.append(roc_auc_RFT);
#     roc_auc_NB_sum.append(roc_auc_NB);roc_auc_SVM_sum.append(roc_auc_SVM);
#
# # print(np.mean(score_KNN_sum));print(np.mean(score_DT_sum));print(np.mean(score_RFT_sum));print(np.mean(score_NB_sum));
# # print(np.mean(score_SVM_sum));
#
# # print(score_KNN_sum);
# # print(len(score_KNN_sum));
# # print(score_KNN_sum[0]);
#
# for n in range(len(score_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=score_KNN_sum[n]
#     df['DT']=score_DT_sum[n]
#     df['RFT']=score_RFT_sum[n]
#     df['NB']=score_NB_sum[n]
#     df['SVM']=score_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\allchannels_June\\statistic_raw_60_45min_channels_ECGperformance_Accuracy_test.csv')
#
#
# # for n in range(len(roc_auc_KNN_sum)):
# #     df = pd.DataFrame()
# #     df['KNN']=roc_auc_KNN_sum[n]
# #     df['DT']=roc_auc_DT_sum[n]
# #     df['RFT']=roc_auc_RFT_sum[n]
# #     df['NB']=roc_auc_NB_sum[n]
# #     df['SVM']=roc_auc_SVM_sum[n]
# #     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\allchannels_June\\statistic_raw_60_45min_channels_EEGperformance_AUC.csv')




















# channels = ['6hFz_EEGvar', '6hF3_EEGvar', '6hF4_EEGvar', '6hF7_EEGvar', '6hF8_EEGvar', '6hC4_EEGvar', '6hC3_EEGvar',
#             '6hCz_EEGvar',
#             '6hPz_EEGvar', '6hP3_EEGvar', '6hP4_EEGvar', '6hT3_EEGvar', '6hT4_EEGvar', '6hT5_EEGvar', '6hT6_EEGvar',
#             '6hO1_EEGvar',
#             '6hO2_EEGvar',
#
#             '6hFz_EEGauto', '6hF3_EEGauto', '6hF4_EEGauto', '6hF7_EEGauto', '6hF8_EEGauto', '6hC4_EEGauto',
#             '6hC3_EEGauto', '6hCz_EEGauto',
#             '6hPz_EEGauto', '6hP3_EEGauto', '6hP4_EEGauto', '6hT3_EEGauto', '6hT4_EEGauto', '6hT5_EEGauto',
#             '6hT6_EEGauto', '6hO1_EEGauto',
#             '6hO2_EEGauto',
#
#             'type',
#             ]
#
#
# score_KNN_sum = []; recall_KNN = []; F1_KNN = [];roc_auc_KNN_sum=[]
# score_DT_sum = []; recall_DT = []; F1_DT = [];roc_auc_DT_sum=[]
# score_RFT_sum = []; recall_RFT = []; F1_RFT = [];roc_auc_RFT_sum=[]
# score_NB_sum = []; recall_NB = []; F1_NB = [];roc_auc_NB_sum=[]
# score_SVM_sum = []; recall_SVM = []; F1_SVM = []; roc_auc_SVM_sum=[]
#
# for m in range(1):
#     score_KNN = [];score_DT = [];score_RFT = [];score_NB = [];score_SVM = [];
#     roc_auc_KNN=[];roc_auc_DT = [];roc_auc_RFT = [];roc_auc_NB = [];roc_auc_SVM = [];
#     for i in range(100):
#         ### evaluate algorithms
#         dataset = pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/PNES_preictal_and_ES_preictal(30_15min).csv',sep=',')
#         column_name = dataset.columns[6:].tolist()
#         print(column_name)
#
#         df=dataset[channels]
#         df=df.dropna()
#
#         class_count_0, class_count_1 = df['type'].value_counts()
#         print(class_count_0); print(class_count_1)
#         class_0 = df[df['type'] == 1]
#         class_1 = df[df['type'] == 0]
#         print('class 0:', class_0.shape);
#         print('class 1:', class_1.shape);
#         class_0_under = class_0.sample(class_count_1)
#         test_under = pd.concat([class_0_under, class_1], axis=0)
#
#
#         X = test_under[[channels[m],channels[m+17]]]
#
#         y = test_under[channels[-1]]
#
#
#         X_train, X_validation, Y_train, Y_validation = train_test_split(X, y.values.ravel(), test_size=0.20, random_state=1)
#         X_train_1 = tf.keras.utils.normalize(X_train, axis=1)
#         X_validation_1 = tf.keras.utils.normalize(X_validation, axis=1)
#
#         # # Make predictions on validation dataset
#         model = KNeighborsClassifier()
#         model.fit(X_train_1, Y_train)
#         predictions = model.predict(X_validation_1)
#         score_KNN.append(accuracy_score(Y_validation, predictions))
#         # recall_KNN.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_KNN.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_KNN.append(roc_auc_score(Y_validation, predictions))
#
#         model = DecisionTreeClassifier()
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_DT.append(accuracy_score(Y_validation, predictions))
#         # recall_DT.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_DT.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_DT.append(roc_auc_score(Y_validation, predictions))
#
#         model = RandomForestClassifier(n_estimators=100)
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_RFT.append(accuracy_score(Y_validation, predictions))
#         # recall_RFT.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_RFT.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_RFT.append(roc_auc_score(Y_validation, predictions))
#
#         model = GaussianNB()
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_NB.append(accuracy_score(Y_validation, predictions))
#         # recall_NB.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_NB.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_NB.append(roc_auc_score(Y_validation, predictions))
#
#
#         model = SVC(gamma='auto')
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_SVM.append(accuracy_score(Y_validation, predictions))
#         # recall_SVM.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_SVM.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_SVM.append(roc_auc_score(Y_validation, predictions))
#
#     score_KNN_sum.append(score_KNN);score_DT_sum.append(score_DT);score_RFT_sum.append(score_RFT);
#     score_NB_sum.append(score_NB);score_SVM_sum.append(score_SVM);
#     roc_auc_KNN_sum.append(roc_auc_KNN);roc_auc_DT_sum.append(roc_auc_DT);roc_auc_RFT_sum.append(roc_auc_RFT);
#     roc_auc_NB_sum.append(roc_auc_NB);roc_auc_SVM_sum.append(roc_auc_SVM);
#
# # print(np.mean(score_KNN_sum));print(np.mean(score_DT_sum));print(np.mean(score_RFT_sum));print(np.mean(score_NB_sum));
# # print(np.mean(score_SVM_sum));
# for n in range(len(score_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=score_KNN_sum[n]
#     df['DT']=score_DT_sum[n]
#     df['RFT']=score_RFT_sum[n]
#     df['NB']=score_NB_sum[n]
#     df['SVM']=score_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\allchannels_June\\onechannel\\6h_30_15min_ALLchannels_EEGperformance_Accuracy_{n}.csv')
#
#
# for n in range(len(roc_auc_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=roc_auc_KNN_sum[n]
#     df['DT']=roc_auc_DT_sum[n]
#     df['RFT']=roc_auc_RFT_sum[n]
#     df['NB']=roc_auc_NB_sum[n]
#     df['SVM']=roc_auc_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\allchannels_June\\onechannel\\6h_30_15min_ALLchannels_EEGperformance_AUC_{n}.csv')







# channels = ['Fz_EEGvar', 'C4_EEGvar', 'Pz_EEGvar', 'C3_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'P4_EEGvar',
#             'P3_EEGvar', 'T4_EEGvar', 'T3_EEGvar', 'O2_EEGvar', 'O1_EEGvar', 'F7_EEGvar', 'F8_EEGvar',
#             'T6_EEGvar', 'T5_EEGvar', 'Cz_EEGvar',
#             'Fz_EEGauto', 'C4_EEGauto', 'Pz_EEGauto', 'C3_EEGauto', 'F3_EEGauto', 'F4_EEGauto', 'P4_EEGauto',
#             'P3_EEGauto', 'T4_EEGauto', 'T3_EEGauto', 'O2_EEGauto', 'O1_EEGauto', 'F7_EEGauto', 'F8_EEGauto',
#             'T6_EEGauto', 'T5_EEGauto', 'Cz_EEGauto']
#
# score_KNN_sum = []; recall_KNN = []; F1_KNN = [];roc_auc_KNN_sum=[]
# score_DT_sum = []; recall_DT = []; F1_DT = [];roc_auc_DT_sum=[]
# score_RFT_sum = []; recall_RFT = []; F1_RFT = [];roc_auc_RFT_sum=[]
# score_NB_sum = []; recall_NB = []; F1_NB = [];roc_auc_NB_sum=[]
# score_SVM_sum = []; recall_SVM = []; F1_SVM = []; roc_auc_SVM_sum=[]
#
# for m in range(17):
#     score_KNN = [];score_DT = [];score_RFT = [];score_NB = [];score_SVM = [];
#     roc_auc_KNN=[];roc_auc_DT = [];roc_auc_RFT = [];roc_auc_NB = [];roc_auc_SVM = [];
#     # for i in range(100):
#     for i in range(100):
#         ### evaluate algorithms
#         dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/PNES_and_ES_preictal.csv',sep=',')
#         class_count_0,class_count_1,class_count_2= dataset['lobe'].value_counts()
#         # print(class_count_0);print(class_count_1);print(class_count_2);
#         class_0 = dataset[dataset['lobe'] == 'T']
#         class_1 = dataset[dataset['lobe'] == 'P']
#         class_2 = dataset[dataset['lobe'] == 'G']
#         # print('class 0:', class_0.shape);
#         # print('class 1:', class_1.shape);
#         # print('class 2:', class_2.shape);
#         class_0_under = class_0.sample(class_count_1)
#         test_under = pd.concat([class_0_under, class_1], axis=0)
#         # print(test_under)
#
#         X = test_under[[channels[m], channels[m + 17]]]
#         # X = test_under[[channels[m],channels[m+17],'ch31_RRIvar', 'ch31_RRIauto']]
#         # X = test_under[['ch31_RRIvar', 'ch31_RRIauto']]
#
#         y = test_under[['lobe']]
#         y=y.copy()
#         y['lobe']=pd.get_dummies(y['lobe'])
#
#         X_train, X_validation, Y_train, Y_validation = train_test_split(X, y.values.ravel(), test_size=0.20, random_state=1)
#         X_train_1 = tf.keras.utils.normalize(X_train, axis=1)
#         X_validation_1 = tf.keras.utils.normalize(X_validation, axis=1)
#
#         # # Make predictions on validation dataset
#         model = KNeighborsClassifier()
#         model.fit(X_train_1, Y_train)
#         predictions = model.predict(X_validation_1)
#         score_KNN.append(accuracy_score(Y_validation, predictions))
#         recall_KNN.append(recall_score(Y_validation, predictions, average='weighted'))
#         F1_KNN.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_KNN.append(roc_auc_score(Y_validation, predictions))
#
#         model = DecisionTreeClassifier()
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_DT.append(accuracy_score(Y_validation, predictions))
#         recall_DT.append(recall_score(Y_validation, predictions, average='weighted'))
#         F1_DT.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_DT.append(roc_auc_score(Y_validation, predictions))
#
#
#         model = RandomForestClassifier(n_estimators=100)
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_RFT.append(accuracy_score(Y_validation, predictions))
#         recall_RFT.append(recall_score(Y_validation, predictions, average='weighted'))
#         F1_RFT.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_RFT.append(roc_auc_score(Y_validation, predictions))
#
#         model = GaussianNB()
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_NB.append(accuracy_score(Y_validation, predictions))
#         recall_NB.append(recall_score(Y_validation, predictions, average='weighted'))
#         F1_NB.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_NB.append(roc_auc_score(Y_validation, predictions))
#
#         model = SVC(gamma='auto')
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_SVM.append(accuracy_score(Y_validation, predictions))
#         recall_SVM.append(recall_score(Y_validation, predictions, average='weighted'))
#         F1_SVM.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_SVM.append(roc_auc_score(Y_validation, predictions))
#
#
#     score_KNN_sum.append(score_KNN);score_DT_sum.append(score_DT);score_RFT_sum.append(score_RFT);
#     score_NB_sum.append(score_NB);score_SVM_sum.append(score_SVM);
#     roc_auc_KNN_sum.append(roc_auc_KNN);roc_auc_DT_sum.append(roc_auc_DT);roc_auc_RFT_sum.append(roc_auc_RFT);
#     roc_auc_NB_sum.append(roc_auc_NB);roc_auc_SVM_sum.append(roc_auc_SVM);
#
# for n in range(len(score_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=score_KNN_sum[n]
#     df['DT']=score_DT_sum[n]
#     df['RFT']=score_RFT_sum[n]
#     df['NB']=score_NB_sum[n]
#     df['SVM']=score_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\Temporal_EEGperformance_Accuracy_{n}.csv')
#
# for n in range(len(roc_auc_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=roc_auc_KNN_sum[n]
#     df['DT']=roc_auc_DT_sum[n]
#     df['RFT']=roc_auc_RFT_sum[n]
#     df['NB']=roc_auc_NB_sum[n]
#     df['SVM']=roc_auc_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\Temporal_EEGperformance_AUC_{n}.csv')







# ### classfication ictal data using cs features
# channels = ['Fz_EEGvar', 'C4_EEGvar', 'Pz_EEGvar', 'C3_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'P4_EEGvar',
#             'P3_EEGvar', 'T4_EEGvar', 'T3_EEGvar', 'O2_EEGvar', 'O1_EEGvar', 'F7_EEGvar', 'F8_EEGvar',
#             'T6_EEGvar', 'T5_EEGvar', 'Cz_EEGvar',
#             'Fz_EEGauto', 'C4_EEGauto', 'Pz_EEGauto', 'C3_EEGauto', 'F3_EEGauto', 'F4_EEGauto', 'P4_EEGauto',
#             'P3_EEGauto', 'T4_EEGauto', 'T3_EEGauto', 'O2_EEGauto', 'O1_EEGauto', 'F7_EEGauto', 'F8_EEGauto',
#             'T6_EEGauto', 'T5_EEGauto', 'Cz_EEGauto']
#
# score_KNN_sum = []; recall_KNN = []; F1_KNN = [];roc_auc_KNN_sum=[]
# score_DT_sum = []; recall_DT = []; F1_DT = [];roc_auc_DT_sum=[]
# score_RFT_sum = []; recall_RFT = []; F1_RFT = [];roc_auc_RFT_sum=[]
# score_NB_sum = []; recall_NB = []; F1_NB = [];roc_auc_NB_sum=[]
# score_SVM_sum = []; recall_SVM = []; F1_SVM = []; roc_auc_SVM_sum=[]
#
# for m in range(1):
#     score_KNN = [];score_DT = [];score_RFT = [];score_NB = [];score_SVM = [];
#     roc_auc_KNN=[];roc_auc_DT = [];roc_auc_RFT = [];roc_auc_NB = [];roc_auc_SVM = [];
#     # for i in range(1):
#     for i in range(100):
#         ### evaluate algorithms
#         dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/PNES_and_ES_ictal.csv',sep=',')
#         class_count_0, class_count_1 = dataset['type'].value_counts()
#         # print(class_count_0); print(class_count_1)
#         class_0 = dataset[dataset['type'] == 1] ### epileptic
#         class_1 = dataset[dataset['type'] == 0] ### PNES
#
#         # print('class 0:', class_0.shape);
#         # print('class 1:', class_1.shape);
#
#
#         class_0_under = class_0.sample(class_count_1)
#         test_under = pd.concat([class_0_under, class_1], axis=0)
#         # print(test_under)
#
#         # X = test_under[['Fz_EEGvar', 'C4_EEGvar', 'Pz_EEGvar', 'C3_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'P4_EEGvar',
#         #     'P3_EEGvar', 'T4_EEGvar', 'T3_EEGvar', 'O2_EEGvar', 'O1_EEGvar', 'F7_EEGvar', 'F8_EEGvar',
#         #     'T6_EEGvar', 'T5_EEGvar', 'Cz_EEGvar',
#         #     'Fz_EEGauto', 'C4_EEGauto', 'Pz_EEGauto', 'C3_EEGauto', 'F3_EEGauto', 'F4_EEGauto', 'P4_EEGauto',
#         #     'P3_EEGauto', 'T4_EEGauto', 'T3_EEGauto', 'O2_EEGauto', 'O1_EEGauto', 'F7_EEGauto', 'F8_EEGauto',
#         #     'T6_EEGauto', 'T5_EEGauto', 'Cz_EEGauto']]
#         # X = test_under[[channels[m], channels[m + 17]]]
#         # X = test_under[[channels[m],channels[m+17],'ch31_RRIvar', 'ch31_RRIauto']]
#         X = test_under[['ch31_RRIvar', 'ch31_RRIauto']]
#
#         y = test_under[['type']]
#         y=y.copy()
#         y['type']=pd.get_dummies(y['type'])
#
#         X_train, X_validation, Y_train, Y_validation = train_test_split(X, y.values.ravel(), test_size=0.20, random_state=1)
#         X_train_1 = tf.keras.utils.normalize(X_train, axis=1)
#         X_validation_1 = tf.keras.utils.normalize(X_validation, axis=1)
#
#         # # Make predictions on validation dataset
#         model = KNeighborsClassifier()
#         model.fit(X_train_1, Y_train)
#         predictions = model.predict(X_validation_1)
#         score_KNN.append(accuracy_score(Y_validation, predictions))
#         recall_KNN.append(recall_score(Y_validation, predictions, average='weighted'))
#         F1_KNN.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_KNN.append(roc_auc_score(Y_validation, predictions))
#
#         model = DecisionTreeClassifier()
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_DT.append(accuracy_score(Y_validation, predictions))
#         recall_DT.append(recall_score(Y_validation, predictions, average='weighted'))
#         F1_DT.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_DT.append(roc_auc_score(Y_validation, predictions))
#
#         model = RandomForestClassifier(n_estimators=100)
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_RFT.append(accuracy_score(Y_validation, predictions))
#         recall_RFT.append(recall_score(Y_validation, predictions, average='weighted'))
#         F1_RFT.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_RFT.append(roc_auc_score(Y_validation, predictions))
#
#         model = GaussianNB()
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_NB.append(accuracy_score(Y_validation, predictions))
#         recall_NB.append(recall_score(Y_validation, predictions, average='weighted'))
#         F1_NB.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_NB.append(roc_auc_score(Y_validation, predictions))
#
#
#         model = SVC(gamma='auto')
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_SVM.append(accuracy_score(Y_validation, predictions))
#         recall_SVM.append(recall_score(Y_validation, predictions, average='weighted'))
#         F1_SVM.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_SVM.append(roc_auc_score(Y_validation, predictions))
#
#     score_KNN_sum.append(score_KNN);score_DT_sum.append(score_DT);score_RFT_sum.append(score_RFT);
#     score_NB_sum.append(score_NB);score_SVM_sum.append(score_SVM);
#     roc_auc_KNN_sum.append(roc_auc_KNN);roc_auc_DT_sum.append(roc_auc_DT);roc_auc_RFT_sum.append(roc_auc_RFT);
#     roc_auc_NB_sum.append(roc_auc_NB);roc_auc_SVM_sum.append(roc_auc_SVM);
#
# for n in range(len(score_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=score_KNN_sum[n]
#     df['DT']=score_DT_sum[n]
#     df['RFT']=score_RFT_sum[n]
#     df['NB']=score_NB_sum[n]
#     df['SVM']=score_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\ictal_classify\\ALL_ECGperformance_Accuracy_{n}.csv')
#
# for n in range(len(roc_auc_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=roc_auc_KNN_sum[n]
#     df['DT']=roc_auc_DT_sum[n]
#     df['RFT']=roc_auc_RFT_sum[n]
#     df['NB']=roc_auc_NB_sum[n]
#     df['SVM']=roc_auc_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\ictal_classify\\ALL_ECGperformance_AUC_{n}.csv')



# ## classfiy types of seizures with PNES
# channels = ['Fz_EEGvar', 'C4_EEGvar', 'Pz_EEGvar', 'C3_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'P4_EEGvar',
#             'P3_EEGvar', 'T4_EEGvar', 'T3_EEGvar', 'O2_EEGvar', 'O1_EEGvar', 'F7_EEGvar', 'F8_EEGvar',
#             'T6_EEGvar', 'T5_EEGvar', 'Cz_EEGvar',
#             'Fz_EEGauto', 'C4_EEGauto', 'Pz_EEGauto', 'C3_EEGauto', 'F3_EEGauto', 'F4_EEGauto', 'P4_EEGauto',
#             'P3_EEGauto', 'T4_EEGauto', 'T3_EEGauto', 'O2_EEGauto', 'O1_EEGauto', 'F7_EEGauto', 'F8_EEGauto',
#             'T6_EEGauto', 'T5_EEGauto', 'Cz_EEGauto']
#
# score_KNN_sum = []; recall_KNN = []; F1_KNN = [];roc_auc_KNN_sum=[]
# score_DT_sum = []; recall_DT = []; F1_DT = [];roc_auc_DT_sum=[]
# score_RFT_sum = []; recall_RFT = []; F1_RFT = [];roc_auc_RFT_sum=[]
# score_NB_sum = []; recall_NB = []; F1_NB = [];roc_auc_NB_sum=[]
# score_SVM_sum = []; recall_SVM = []; F1_SVM = []; roc_auc_SVM_sum=[]
#
# for m in range(17):
#     score_KNN = [];score_DT = [];score_RFT = [];score_NB = [];score_SVM = [];
#     roc_auc_KNN=[];roc_auc_DT = [];roc_auc_RFT = [];roc_auc_NB = [];roc_auc_SVM = [];
#     # for i in range(100):
#     for i in range(100):
#         ### evaluate algorithms
#         dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/PNES_and_ES_ictal.csv',sep=',')
#         class_count_0,class_count_1,class_count_2= dataset['lobe'].value_counts()
#         # print(class_count_0);print(class_count_1);print(class_count_2);
#         class_0 = dataset[dataset['lobe'] == 'T']
#         class_1 = dataset[dataset['lobe'] == 'P']
#         class_2 = dataset[dataset['lobe'] == 'G']
#         # print('class 0:', class_0.shape);
#         # print('class 1:', class_1.shape);
#         # print('class 2:', class_2.shape);
#         class_1_under = class_1.sample(class_count_2)
#         test_under = pd.concat([class_1_under, class_2], axis=0)
#         # print(test_under)
#
#         # X = test_under[[channels[m], channels[m + 17]]]
#         X = test_under[[channels[m],channels[m+17],'ch31_RRIvar', 'ch31_RRIauto']]
#         # X = test_under[['ch21_RRIvar', 'ch21_RRIauto']]
#         # print(channels[m]);print(channels[m+17])
#
#         y = test_under[['lobe']]
#         y=y.copy()
#         y['lobe']=pd.get_dummies(y['lobe'])
#
#         # print(y.values.ravel());
#         # print(y),print(len(y));print(X);print(len(X))
#         X_train, X_validation, Y_train, Y_validation = train_test_split(X, y.values.ravel(), test_size=0.20, random_state=1)
#         X_train_1 = tf.keras.utils.normalize(X_train, axis=1)
#         X_validation_1 = tf.keras.utils.normalize(X_validation, axis=1)
#
#         # # Make predictions on validation dataset
#         model = KNeighborsClassifier()
#         model.fit(X_train_1, Y_train)
#         predictions = model.predict(X_validation_1)
#         score_KNN.append(accuracy_score(Y_validation, predictions))
#         recall_KNN.append(recall_score(Y_validation, predictions, average='weighted'))
#         F1_KNN.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_KNN.append(roc_auc_score(Y_validation, predictions))
#
#
#         model = DecisionTreeClassifier()
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_DT.append(accuracy_score(Y_validation, predictions))
#         recall_DT.append(recall_score(Y_validation, predictions, average='weighted'))
#         F1_DT.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_DT.append(roc_auc_score(Y_validation, predictions))
#
#
#         model = RandomForestClassifier(n_estimators=100)
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_RFT.append(accuracy_score(Y_validation, predictions))
#         recall_RFT.append(recall_score(Y_validation, predictions, average='weighted'))
#         F1_RFT.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_RFT.append(roc_auc_score(Y_validation, predictions))
#
#
#         model = GaussianNB()
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_NB.append(accuracy_score(Y_validation, predictions))
#         recall_NB.append(recall_score(Y_validation, predictions, average='weighted'))
#         F1_NB.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_NB.append(roc_auc_score(Y_validation, predictions))
#
#
#
#         model = SVC(gamma='auto')
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_SVM.append(accuracy_score(Y_validation, predictions))
#         recall_SVM.append(recall_score(Y_validation, predictions, average='weighted'))
#         F1_SVM.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_SVM.append(roc_auc_score(Y_validation, predictions))
#
#     score_KNN_sum.append(score_KNN);score_DT_sum.append(score_DT);score_RFT_sum.append(score_RFT);
#     score_NB_sum.append(score_NB);score_SVM_sum.append(score_SVM);
#     roc_auc_KNN_sum.append(roc_auc_KNN);roc_auc_DT_sum.append(roc_auc_DT);roc_auc_RFT_sum.append(roc_auc_RFT);
#     roc_auc_NB_sum.append(roc_auc_NB);roc_auc_SVM_sum.append(roc_auc_SVM);
#
# for n in range(len(score_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=score_KNN_sum[n]
#     df['DT']=score_DT_sum[n]
#     df['RFT']=score_RFT_sum[n]
#     df['NB']=score_NB_sum[n]
#     df['SVM']=score_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\ictal_classify\\Generalise_EEGECGperformance_Accuracy_{n}.csv')
#
# for n in range(len(roc_auc_KNN_sum)):
#     df = pd.DataFrame()
#     df['KNN']=roc_auc_KNN_sum[n]
#     df['DT']=roc_auc_DT_sum[n]
#     df['RFT']=roc_auc_RFT_sum[n]
#     df['NB']=roc_auc_NB_sum[n]
#     df['SVM']=roc_auc_SVM_sum[n]
#     df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\ictal_classify\\Generalise_EEGECGperformance__AUC_{n}.csv')


















