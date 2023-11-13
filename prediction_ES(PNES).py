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


## predict ES (classfiy preictal and interictal data in ES)

channels = ['Fz_EEGvar', 'C4_EEGvar', 'Pz_EEGvar', 'C3_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'P4_EEGvar',
            'P3_EEGvar', 'T4_EEGvar', 'T3_EEGvar', 'O2_EEGvar', 'O1_EEGvar', 'F7_EEGvar', 'F8_EEGvar',
            'T6_EEGvar', 'T5_EEGvar', 'Cz_EEGvar',
            'Fz_EEGauto', 'C4_EEGauto', 'Pz_EEGauto', 'C3_EEGauto', 'F3_EEGauto', 'F4_EEGauto', 'P4_EEGauto',
            'P3_EEGauto', 'T4_EEGauto', 'T3_EEGauto', 'O2_EEGauto', 'O1_EEGauto', 'F7_EEGauto', 'F8_EEGauto',
            'T6_EEGauto', 'T5_EEGauto', 'Cz_EEGauto']

score_KNN_sum = []; recall_KNN = []; F1_KNN = [];roc_auc_KNN_sum=[]
score_DT_sum = []; recall_DT = []; F1_DT = [];roc_auc_DT_sum=[]
score_RFT_sum = []; recall_RFT = []; F1_RFT = [];roc_auc_RFT_sum=[]
score_NB_sum = []; recall_NB = []; F1_NB = [];roc_auc_NB_sum=[]
score_SVM_sum = []; recall_SVM = []; F1_SVM = []; roc_auc_SVM_sum=[]
for m in range(1):
    score_KNN = [];score_DT = [];score_RFT = [];score_NB = [];score_SVM = [];
    roc_auc_KNN=[];roc_auc_DT = [];roc_auc_RFT = [];roc_auc_NB = [];roc_auc_SVM = [];
    for i in range(100):
        dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_45min/ES_interictal_preictal_average_1min.csv',sep=',')
        class_count_0, class_count_1 = dataset['tags'].value_counts()
        # print(class_count_0); print(class_count_1)
        class_0 = dataset[dataset['tags'] == 0]
        class_1 = dataset[dataset['tags'] == 2]
        # print('class 0:', class_0.shape);
        # print('class 1:', class_1.shape);

        class_0_under = class_0.sample(class_count_1)
        test_under = pd.concat([class_0_under, class_1], axis=0)
        # print(len(test_under));print(type(test_under));
        # print(test_under);

        # all channels together m=1
        X = test_under[['Fz_EEGvar', 'C4_EEGvar', 'Pz_EEGvar', 'C3_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'P4_EEGvar',
            'P3_EEGvar', 'T4_EEGvar', 'T3_EEGvar', 'O2_EEGvar', 'O1_EEGvar', 'F7_EEGvar', 'F8_EEGvar',
            'T6_EEGvar', 'T5_EEGvar', 'Cz_EEGvar',
            'Fz_EEGauto', 'C4_EEGauto', 'Pz_EEGauto', 'C3_EEGauto', 'F3_EEGauto', 'F4_EEGauto', 'P4_EEGauto',
            'P3_EEGauto', 'T4_EEGauto', 'T3_EEGauto', 'O2_EEGauto', 'O1_EEGauto', 'F7_EEGauto', 'F8_EEGauto',
            'T6_EEGauto', 'T5_EEGauto', 'Cz_EEGauto']]
        # each channel for each run set m=17
        # X = test_under[[channels[m], channels[m + 17], 'ch31_RRIvar','ch31_RRIauto']]

        # select several channels m=1
        # X = test_under[['Fz_EEGvar', 'F3_EEGvar', 'F4_EEGvar',
        #             'Fz_EEGauto', 'F3_EEGauto', 'F4_EEGauto']]
        # X = test_under[['Cz_EEGvar', 'C3_EEGvar', 'C4_EEGvar',
        #                 'Cz_EEGauto', 'C3_EEGauto', 'C4_EEGauto']]
        # X = test_under[['Pz_EEGvar', 'P3_EEGvar', 'P4_EEGvar',
        #                 'Pz_EEGauto', 'P3_EEGauto', 'P4_EEGauto']]
        # X = test_under[['Fz_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'F7_EEGvar', 'F8_EEGvar',
        #             'Fz_EEGauto', 'F3_EEGauto', 'F4_EEGauto','F7_EEGauto', 'F8_EEGauto']]
        # X = test_under[['T5_EEGvar','T6_EEGvar', 'T3_EEGvar', 'T4_EEGvar',
        #                 'T5_EEGauto','T6_EEGauto', 'T3_EEGauto', 'T4_EEGauto']]
        # X = test_under[['O1_EEGvar', 'O2_EEGvar',
        #                 'O1_EEGauto', 'O2_EEGauto']]

        y = test_under[['tags']]
        y = y.astype('int')

        X_train, X_validation, Y_train, Y_validation = train_test_split(X, y.values.ravel(), test_size=0.20, random_state=1)

        # print(X_train.shape);print(type(X_train));
        # # print(X_train);

        X_train_1 = tf.keras.utils.normalize(X_train)
        X_validation_1 = tf.keras.utils.normalize(X_validation, axis=-1)
        # print(X_train_1);

        model = KNeighborsClassifier()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_KNN.append(accuracy_score(Y_validation, predictions))
        # recall_KNN.append(recall_score(Y_validation, predictions, average='weighted'))
        # F1_KNN.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_KNN.append(roc_auc_score(Y_validation, predictions))

        model = DecisionTreeClassifier()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_DT.append(accuracy_score(Y_validation, predictions))
        recall_DT.append(recall_score(Y_validation, predictions, average='weighted'))
        F1_DT.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_DT.append(roc_auc_score(Y_validation, predictions))

        # model = RandomForestClassifier(n_estimators=100)
        model = RandomForestClassifier(n_estimators=150)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_RFT.append(accuracy_score(Y_validation, predictions))
        # recall_RFT.append(recall_score(Y_validation, predictions, average='weighted'))
        # F1_RFT.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_RFT.append(roc_auc_score(Y_validation, predictions))

        model = GaussianNB()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_NB.append(accuracy_score(Y_validation, predictions))
        # recall_NB.append(recall_score(Y_validation, predictions, average='weighted'))
        # F1_NB.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_NB.append(roc_auc_score(Y_validation, predictions))


        model = SVC(gamma='auto')
        # model = SVC(gamma='scale')
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_SVM.append(accuracy_score(Y_validation, predictions))
        # recall_SVM.append(recall_score(Y_validation, predictions, average='weighted'))
        # F1_SVM.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_SVM.append(roc_auc_score(Y_validation, predictions))

    score_KNN_sum.append(score_KNN);score_DT_sum.append(score_DT);score_RFT_sum.append(score_RFT);
    score_NB_sum.append(score_NB);score_SVM_sum.append(score_SVM);
    roc_auc_KNN_sum.append(roc_auc_KNN);roc_auc_DT_sum.append(roc_auc_DT);roc_auc_RFT_sum.append(roc_auc_RFT);
    roc_auc_NB_sum.append(roc_auc_NB);roc_auc_SVM_sum.append(roc_auc_SVM);

print(np.mean(roc_auc_KNN_sum));
print(np.mean(roc_auc_DT_sum));
print(np.mean(roc_auc_RFT_sum));
print(np.mean(roc_auc_NB_sum));
print(np.mean(roc_auc_SVM_sum));

for n in range(len(score_KNN_sum)):
    channel=channels[n]
    df = pd.DataFrame()
    df['KNN']=score_KNN_sum[n]
    df['DT']=score_DT_sum[n]
    df['RFT']=score_RFT_sum[n]
    df['NB']=score_NB_sum[n]
    df['SVM']=score_SVM_sum[n]
    df.to_csv(f'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_45min/performance/classifiers_EEGperformance_Accuracy_All_1min.csv')

for n in range(len(roc_auc_KNN_sum)):
    channel=channels[n]
    df = pd.DataFrame()
    df['KNN']=roc_auc_KNN_sum[n]
    df['DT']=roc_auc_DT_sum[n]
    df['RFT']=roc_auc_RFT_sum[n]
    df['NB']=roc_auc_NB_sum[n]
    df['SVM']=roc_auc_SVM_sum[n]
    df.to_csv(f'C:/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_45min/performance/classifiers_EEGperformance_AUC_All_1min.csv')








# ## predict PNES (classfiy preictal and interictal data in PNES)
#
# channels = ['Fz_EEGvar', 'C4_EEGvar', 'Pz_EEGvar', 'C3_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'P4_EEGvar',
#             'P3_EEGvar', 'T4_EEGvar', 'T3_EEGvar', 'O2_EEGvar', 'O1_EEGvar', 'F7_EEGvar', 'F8_EEGvar',
#             'T6_EEGvar', 'T5_EEGvar', 'Cz_EEGvar',
#             'Fz_EEGauto', 'C4_EEGauto', 'Pz_EEGauto', 'C3_EEGauto', 'F3_EEGauto', 'F4_EEGauto', 'P4_EEGauto',
#             'P3_EEGauto', 'T4_EEGauto', 'T3_EEGauto', 'O2_EEGauto', 'O1_EEGauto', 'F7_EEGauto', 'F8_EEGauto',
#             'T6_EEGauto', 'T5_EEGauto', 'Cz_EEGauto']
#
# score_KNN_sum = [];
# recall_KNN = [];
# F1_KNN = [];
# roc_auc_KNN_sum = []
# score_DT_sum = [];
# recall_DT = [];
# F1_DT = [];
# roc_auc_DT_sum = []
# score_RFT_sum = [];
# recall_RFT = [];
# F1_RFT = [];
# roc_auc_RFT_sum = []
# score_NB_sum = [];
# recall_NB = [];
# F1_NB = [];
# roc_auc_NB_sum = []
# score_SVM_sum = [];
# recall_SVM = [];
# F1_SVM = [];
# roc_auc_SVM_sum = []
# for m in range(1):
#     score_KNN = [];
#     score_DT = [];
#     score_RFT = [];
#     score_NB = [];
#     score_SVM = [];
#     roc_auc_KNN = [];
#     roc_auc_DT = [];
#     roc_auc_RFT = [];
#     roc_auc_NB = [];
#     roc_auc_SVM = [];
#     for i in range(100):
#         dataset = pd.read_csv(
#             'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags_30min/PNES_interictal_preictal_average.csv',
#             sep=',')
#         class_count_0, class_count_1 = dataset['tags'].value_counts()
#         # print(class_count_0); print(class_count_1)
#         class_0 = dataset[dataset['tags'] == 0]
#         class_1 = dataset[dataset['tags'] == 2]
#         # print('class 0:', class_0.shape);
#         # print('class 1:', class_1.shape);
#
#         class_0_under = class_0.sample(class_count_1)
#         test_under = pd.concat([class_0_under, class_1], axis=0)
#         # print(len(test_under));print(type(test_under));
#         # print(test_under);
#
#         # all channels together m=1
#         X = test_under[['Fz_EEGvar', 'C4_EEGvar', 'Pz_EEGvar', 'C3_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'P4_EEGvar',
#                         'P3_EEGvar', 'T4_EEGvar', 'T3_EEGvar', 'O2_EEGvar', 'O1_EEGvar', 'F7_EEGvar', 'F8_EEGvar',
#                         'T6_EEGvar', 'T5_EEGvar', 'Cz_EEGvar',
#                         'Fz_EEGauto', 'C4_EEGauto', 'Pz_EEGauto', 'C3_EEGauto', 'F3_EEGauto', 'F4_EEGauto',
#                         'P4_EEGauto',
#                         'P3_EEGauto', 'T4_EEGauto', 'T3_EEGauto', 'O2_EEGauto', 'O1_EEGauto', 'F7_EEGauto',
#                         'F8_EEGauto',
#                         'T6_EEGauto', 'T5_EEGauto', 'Cz_EEGauto']]
#         # each channel for each run set m=17
#         # X = test_under[[channels[m], channels[m + 17], 'ch31_RRIvar','ch31_RRIauto']]
#
#         # select several channels m=1
#         # X = test_under[['Fz_EEGvar', 'F3_EEGvar', 'F4_EEGvar',
#         #             'Fz_EEGauto', 'F3_EEGauto', 'F4_EEGauto']]
#         # X = test_under[['Cz_EEGvar', 'C3_EEGvar', 'C4_EEGvar',
#         #                 'Cz_EEGauto', 'C3_EEGauto', 'C4_EEGauto']]
#         # X = test_under[['Pz_EEGvar', 'P3_EEGvar', 'P4_EEGvar',
#         #                 'Pz_EEGauto', 'P3_EEGauto', 'P4_EEGauto']]
#         # X = test_under[['Fz_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'F7_EEGvar', 'F8_EEGvar',
#         #             'Fz_EEGauto', 'F3_EEGauto', 'F4_EEGauto','F7_EEGauto', 'F8_EEGauto']]
#         # X = test_under[['T5_EEGvar','T6_EEGvar', 'T3_EEGvar', 'T4_EEGvar',
#         #                 'T5_EEGauto','T6_EEGauto', 'T3_EEGauto', 'T4_EEGauto']]
#         # X = test_under[['O1_EEGvar', 'O2_EEGvar',
#         #                 'O1_EEGauto', 'O2_EEGauto']]
#
#         y = test_under[['tags']]
#         y = y.astype('int')
#
#         X_train, X_validation, Y_train, Y_validation = train_test_split(X, y.values.ravel(), test_size=0.20,
#                                                                         random_state=1)
#         # print(X_train.shape);print(type(X_train));
#         # # print(X_train);
#
#         X_train_1 = tf.keras.utils.normalize(X_train)
#         X_validation_1 = tf.keras.utils.normalize(X_validation, axis=-1)
#         # print(X_train_1);
#
#         model = KNeighborsClassifier()
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_KNN.append(accuracy_score(Y_validation, predictions))
#         # recall_KNN.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_KNN.append(f1_score(Y_validation, predictions, average='weighted'))
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
#         model = SVC(gamma='auto')
#         model.fit(X_train, Y_train)
#         predictions = model.predict(X_validation)
#         score_SVM.append(accuracy_score(Y_validation, predictions))
#         # recall_SVM.append(recall_score(Y_validation, predictions, average='weighted'))
#         # F1_SVM.append(f1_score(Y_validation, predictions, average='weighted'))
#         roc_auc_SVM.append(roc_auc_score(Y_validation, predictions))
#
#     score_KNN_sum.append(score_KNN);
#     score_DT_sum.append(score_DT);
#     score_RFT_sum.append(score_RFT);
#     score_NB_sum.append(score_NB);
#     score_SVM_sum.append(score_SVM);
#     roc_auc_KNN_sum.append(roc_auc_KNN);
#     roc_auc_DT_sum.append(roc_auc_DT);
#     roc_auc_RFT_sum.append(roc_auc_RFT);
#     roc_auc_NB_sum.append(roc_auc_NB);
#     roc_auc_SVM_sum.append(roc_auc_SVM);
#
# for n in range(len(score_KNN_sum)):
#     channel = channels[n]
#     df = pd.DataFrame()
#     df['KNN'] = score_KNN_sum[n]
#     df['DT'] = score_DT_sum[n]
#     df['RFT'] = score_RFT_sum[n]
#     df['NB'] = score_NB_sum[n]
#     df['SVM'] = score_SVM_sum[n]
#     df.to_csv(
#         f'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags_30min/performance/classifiers_EEGperformance_Accuracy_All.csv')
#
# for n in range(len(roc_auc_KNN_sum)):
#     channel = channels[n]
#     df = pd.DataFrame()
#     df['KNN'] = roc_auc_KNN_sum[n]
#     df['DT'] = roc_auc_DT_sum[n]
#     df['RFT'] = roc_auc_RFT_sum[n]
#     df['NB'] = roc_auc_NB_sum[n]
#     df['SVM'] = roc_auc_SVM_sum[n]
#     df.to_csv(
#         f'C:/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags_30min/performance/classifiers_EEGperformance_AUC_All.csv')