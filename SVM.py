from __future__ import division
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,f1_score,roc_auc_score,precision_score
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

            'type']


score_KNN_sum = []; recall_KNN_sum = []; F1_KNN_sum = [];roc_auc_KNN_sum=[]; precision_KNN_sum=[]
score_DT_sum = []; recall_DT_sum = []; F1_DT_sum = [];roc_auc_DT_sum=[];precision_DT_sum=[]
score_RFT_sum = []; recall_RFT_sum = []; F1_RFT_sum = [];roc_auc_RFT_sum=[];precision_RFT_sum=[]
score_NB_sum = []; recall_NB_sum = []; F1_NB_sum = [];roc_auc_NB_sum=[];precision_NB_sum=[]
score_SVM_sum = []; recall_SVM_sum = []; F1_SVM_sum = []; roc_auc_SVM_sum=[];precision_SVM_sum=[]
for m in range(1):
    score_KNN = [];score_DT = [];score_RFT = [];score_NB = [];score_SVM = [];
    F1_KNN = [];F1_DT = [];F1_RFT = [];F1_NB = [];F1_SVM = [];
    recall_KNN=[];recall_DT = [];recall_RFT = [];recall_NB = [];recall_SVM = [];
    roc_auc_KNN = [];roc_auc_DT = [];roc_auc_RFT = [];roc_auc_NB = [];roc_auc_SVM = [];
    precision_KNN = [];precision_DT = [];precision_RFT = [];precision_NB = [];precision_SVM = [];
    for i in range(100):
        ### evaluate algorithms
        dataset = pd.read_csv('C:/Users/wxiong/Documents/PHD/combine_features/PNES_preictal_and_ES_preictal(45_30min)_raw.csv',sep=',')
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

        X = test_under[channels[0:-1]]
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
        precision_KNN.append(precision_score(Y_validation, predictions, average='weighted'))

        model = DecisionTreeClassifier()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_DT.append(accuracy_score(Y_validation, predictions))
        recall_DT.append(recall_score(Y_validation, predictions, average='weighted'))
        F1_DT.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_DT.append(roc_auc_score(Y_validation, predictions))
        precision_DT.append(precision_score(Y_validation, predictions, average='weighted'))

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_RFT.append(accuracy_score(Y_validation, predictions))
        recall_RFT.append(recall_score(Y_validation, predictions, average='weighted'))
        F1_RFT.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_RFT.append(roc_auc_score(Y_validation, predictions))
        precision_RFT.append(precision_score(Y_validation, predictions, average='weighted'))

        model = GaussianNB()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_NB.append(accuracy_score(Y_validation, predictions))
        recall_NB.append(recall_score(Y_validation, predictions, average='weighted'))
        F1_NB.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_NB.append(roc_auc_score(Y_validation, predictions))
        precision_NB.append(precision_score(Y_validation, predictions, average='weighted'))

        model = SVC(gamma='auto', kernel='rbf')
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        score_SVM.append(accuracy_score(Y_validation, predictions))
        recall_SVM.append(recall_score(Y_validation, predictions, average='weighted'))
        F1_SVM.append(f1_score(Y_validation, predictions, average='weighted'))
        roc_auc_SVM.append(roc_auc_score(Y_validation, predictions))
        precision_SVM.append(precision_score(Y_validation, predictions, average='weighted'))


    score_KNN_sum.append(score_KNN);score_DT_sum.append(score_DT);score_RFT_sum.append(score_RFT);
    score_NB_sum.append(score_NB);score_SVM_sum.append(score_SVM);
    roc_auc_KNN_sum.append(roc_auc_KNN);roc_auc_DT_sum.append(roc_auc_DT);roc_auc_RFT_sum.append(roc_auc_RFT);
    roc_auc_NB_sum.append(roc_auc_NB);roc_auc_SVM_sum.append(roc_auc_SVM);
    F1_KNN_sum.append(F1_KNN);F1_DT_sum.append(F1_DT);F1_RFT_sum.append(F1_RFT);
    F1_NB_sum.append(F1_NB);F1_SVM_sum.append(F1_SVM);
    recall_KNN_sum.append(recall_KNN);recall_DT_sum.append(recall_DT);recall_RFT_sum.append(recall_RFT);
    recall_NB_sum.append(recall_NB);recall_SVM_sum.append(recall_SVM);
    precision_KNN_sum.append(precision_KNN);precision_DT_sum.append(precision_DT);precision_RFT_sum.append(precision_RFT);
    precision_NB_sum.append(precision_NB);precision_SVM_sum.append(precision_SVM);



for n in range(len(score_KNN_sum)):
    df = pd.DataFrame()
    df['KNN']=score_KNN_sum[n]
    df['DT']=score_DT_sum[n]
    df['RFT']=score_RFT_sum[n]
    df['NB']=score_NB_sum[n]
    df['SVM']=score_SVM_sum[n]
    df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\allchannels_June\\raw_15_0min_channels_EEGperformance_Accuracy.csv')


for n in range(len(roc_auc_KNN_sum)):
    df = pd.DataFrame()
    df['KNN']=roc_auc_KNN_sum[n]
    df['DT']=roc_auc_DT_sum[n]
    df['RFT']=roc_auc_RFT_sum[n]
    df['NB']=roc_auc_NB_sum[n]
    df['SVM']=roc_auc_SVM_sum[n]
    df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\allchannels_June\\raw_120_105min_channels_EEGperformance_AUC.csv')

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

for n in range(len(precision_KNN_sum)):
    df = pd.DataFrame()
    df['KNN'] = precision_KNN_sum[n]
    df['DT'] = precision_DT_sum[n]
    df['RFT'] = precision_RFT_sum[n]
    df['NB'] = precision_NB_sum[n]
    df['SVM'] = precision_SVM_sum[n]
    df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\allchannels_June\\raw_45_30min_channels_EEGECGperformance_precision_x.csv')
