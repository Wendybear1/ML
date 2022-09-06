from __future__ import division
import pandas as pd
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import tensorflow as tf



channels = ['Fz_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'F7_EEGvar', 'F8_EEGvar', 'C4_EEGvar', 'C3_EEGvar', 'Cz_EEGvar',
            'Pz_EEGvar', 'P3_EEGvar', 'P4_EEGvar', 'T3_EEGvar', 'T4_EEGvar', 'T5_EEGvar', 'T6_EEGvar', 'O1_EEGvar',
            'O2_EEGvar',

            'Fz_EEGauto', 'F3_EEGauto', 'F4_EEGauto', 'F7_EEGauto', 'F8_EEGauto', 'C4_EEGauto', 'C3_EEGauto','Cz_EEGauto',
            'Pz_EEGauto', 'P3_EEGauto', 'P4_EEGauto', 'T3_EEGauto', 'T4_EEGauto', 'T5_EEGauto', 'T6_EEGauto','O1_EEGauto',
            'O2_EEGauto',

            'ch31_RRIvar', 'ch31_RRIauto',

            'type',]




score_KNN = [];score_DT = [];score_RFT = [];score_NB = [];score_SVM = [];
F1_KNN = [];F1_DT = [];F1_RFT = [];F1_NB = [];F1_SVM = [];
recall_KNN=[];recall_DT = [];recall_RFT = [];recall_NB = [];recall_SVM = [];
roc_auc_KNN = [];roc_auc_DT = [];roc_auc_RFT = [];roc_auc_NB = [];roc_auc_SVM = [];
for i in range(5):
    # load data
    dataset = pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/PNES_preictal_and_ES_preictal(15_0min)_raw.csv',sep=',')
    column_name = dataset.columns[6:].tolist()
    # drop Nan values
    df=dataset[channels]
    df=df.dropna()
    # count number of each class
    class_count_0, class_count_1 = df['type'].value_counts()
    # make unbalance classes equal
    class_0 = df[df['type'] == 1]
    class_1 = df[df['type'] == 0]
    class_0_under = class_0.sample(class_count_1)
    test_under = pd.concat([class_0_under, class_1], axis=0)

    # split training and testing datasets
    X = test_under[channels[0:-1]]
    y = test_under[channels[-1]]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y.values.ravel(), test_size=0.20, random_state=1)
    X_train_1 = tf.keras.utils.normalize(X_train, axis=1)
    X_validation_1 = tf.keras.utils.normalize(X_validation, axis=1)

    #  Make predictions
    # KNN classifier
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_1, Y_train)
    predictions = model.predict(X_validation_1)
    score_KNN.append(accuracy_score(Y_validation, predictions))
    recall_KNN.append(recall_score(Y_validation, predictions, average='weighted'))
    F1_KNN.append(f1_score(Y_validation, predictions, average='weighted'))

    # Desicion tree classifier
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    score_DT.append(accuracy_score(Y_validation, predictions))
    recall_DT.append(recall_score(Y_validation, predictions, average='weighted'))
    F1_DT.append(f1_score(Y_validation, predictions, average='weighted'))

    # Random forest classifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    score_RFT.append(accuracy_score(Y_validation, predictions))
    recall_RFT.append(recall_score(Y_validation, predictions, average='weighted'))
    F1_RFT.append(f1_score(Y_validation, predictions, average='weighted'))

    # Naive bayesian classifier
    model = GaussianNB()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    score_NB.append(accuracy_score(Y_validation, predictions))
    recall_NB.append(recall_score(Y_validation, predictions, average='weighted'))
    F1_NB.append(f1_score(Y_validation, predictions, average='weighted'))

    # Support vector machine classifier
    model = SVC(gamma='auto', kernel='rbf')
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    score_SVM.append(accuracy_score(Y_validation, predictions))
    recall_SVM.append(recall_score(Y_validation, predictions, average='weighted'))
    F1_SVM.append(f1_score(Y_validation, predictions, average='weighted'))



df = pd.DataFrame()
df['KNN']=score_KNN
df['DT']=score_DT
df['RFT']=score_RFT
df['NB']=score_NB
df['SVM']=score_SVM
df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\allchannels_June\\EEGECGperformance_accuracy.csv')




df = pd.DataFrame()
df['KNN']=F1_KNN
df['DT']=F1_DT
df['RFT']=F1_RFT
df['NB']=F1_NB
df['SVM']=F1_SVM
df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\allchannels_June\\EEGECGperformance_F1.csv')


df = pd.DataFrame()
df['KNN']=recall_KNN
df['DT']=recall_DT
df['RFT']=recall_RFT
df['NB']=recall_NB
df['SVM']=recall_SVM
df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\allchannels_June\\EEGECGperformance_recall.csv')
