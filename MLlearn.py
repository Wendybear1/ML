from __future__ import division
import mne
import numpy as np
import scipy.signal
from matplotlib import pyplot
import math
from scipy.fftpack import fft, ifft
from scipy import signal
from scipy.signal import butter, lfilter,iirfilter
from biosppy.signals import tools
from statsmodels.tsa.ar_model import AutoReg
import scipy.stats as stats
from scipy.stats import norm

import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from scipy.signal import lfilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns



def movingaverage(values, window_size):
    weights = (np.ones(window_size))/window_size
    a=np.ones(1)
    return lfilter(weights,a,values)


# ## TAS0056
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/TAS0056/Cz_EEGvariance_TAS0056_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/TAS0056/Cz_EEGauto_TAS0056_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_variance_EEG_arr))
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/TAS0056/RRI_ch31_rawvariance_TAS0056_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/TAS0056/RRI_ch31_rawauto_TAS0056_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# index_1=[1475,2541,4234,8370,10353,11647,14479,18809,20161,21596,25736,26497,27232,31995,37691,39155] ## leading seizure index
# index_2=[8748,9074,9619,14600,14863,14926,15453,27562,27754,32269,38075,38423] ### seizure cluster index
# index=index_1+index_2
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((39845,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# print(Raw_class_arr[seizure_index[0]]);print(seizure_index[0]);print(seizure_index[1]);
# print(Raw_class_arr[nonseizure_index[0]]);print(nonseizure_index[0]);print(nonseizure_index[1]);
#
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# # print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_PNES_TAS0056_features.csv")

# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_TAS0056_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));
# # print(dataset.describe());
# print(dataset.groupby('Class').size());
# ### data visualization
data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_TAS0056_features.csv',sep=',',usecols=[1,2,3,4])
data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
data.hist()
pyplot.show()
scatter_matrix(data)
pyplot.show()
data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_TAS0056_features.csv',sep=',',usecols=[1,2,3,4,5])
X=data.drop('Class',axis=1)
y=data['Class']
# print(X);print(y)
y=y.astype('int')
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
print(X_train);
# print(X_test);
print(y_train);
# print(y_test);
print(len(y_train));print(len(y_test));
print(sum(y_train));print(sum(y_test));

### Compare Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
print(X_train);print(X_test);
### SVM classifier
clf=svm.SVC(gamma='auto') ### default kernel='rbf',Radial basis function kernel; gamma is same when auto and default scale
clf.fit(X_train,y_train)
pred_clf=clf.predict(X_test)
print(classification_report(y_test,pred_clf))
print(confusion_matrix(y_test,pred_clf))
# from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)
# from sklearn.pipeline import Pipeline
# nca = NeighborhoodComponentsAnalysis(random_state=42)
# knn = KNeighborsClassifier(n_neighbors=3)
# nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
# nca_pipe.fit(X_train, y_train)
# print(nca_pipe.score(X_test, y_test))
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))


# ## QLD0290
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
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_variance_EEG_arr))
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/RRI_ch31_rawvariance_QLD0290_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/RRI_ch31_rawauto_QLD0290_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_RRI31_arr))
# index_1=[5468,10332,11682,15941,22516,28271,29126,33686,34127] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((38995,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# # print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_PNES_QLD0290_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_QLD0290_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# # ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_QLD0290_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_QLD0290_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# ## TAS0058
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/Cz_EEGvariance_TAS0058_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/Cz_EEGauto_TAS0058_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/RRI_ch31_rawvariance_TAS0058_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/RRI_ch31_rawauto_TAS0058_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_RRI31_arr))
# index_1=[3583,5466,7222,8408,9619,11382,17605,21227,21936,23450,27345,29297] ## leading seizure index
# index_2=[7223, 8420, 21852, 29311, 33999]
# index=index_1+index_2
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((38753,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# # print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_PNES_TAS0058_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_TAS0058_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# # ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_TAS0058_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_TAS0058_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))




# ## QLD1230
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1230/Cz_EEGvariance_QLD1230_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1230/Cz_EEGauto_QLD1230_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
#
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1230/RRI_ch31_rawvariance_QLD1230_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1230/RRI_ch31_rawauto_QLD1230_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_RRI31_arr))
#
# index_1=[3252,7014,10660,12255,17875,24345] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((37928,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# # print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_PNES_QLD1230_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_QLD1230_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# # ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_QLD1230_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_QLD1230_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))


# ## VIC1012
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC1012/Cz_EEGvariance_VIC1012_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC1012/Cz_EEGauto_VIC1012_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC1012/RRI_ch21_rawvariance_VIC1012_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC1012/RRI_ch21_rawauto_VIC1012_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_RRI31_arr))
# index_1=[4151,10471,11111,11942,15631,28991] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((36138,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# # print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_PNES_VIC1012_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_VIC1012_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# # ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_VIC1012_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_VIC1012_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
#


# ## VIC0583
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0583/Cz_EEGvariance_VIC0583_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0583/Cz_EEGauto_VIC0583_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
#
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
#
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0583/RRI_ch31_rawvariance_VIC0583_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0583/RRI_ch31_rawauto_VIC0583_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_RRI31_arr))
#
#
# index_1=[4491,5403,6715,8379,12191,13651,17035,18131,19667,24319,26047,31751,35043,35968] ## leading seizure index
# index=index_1
#
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
#
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
#
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
#
# class_arr=np.zeros((40160,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
#
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# # print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_PNES_VIC0583_features.csv")
#
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_VIC0583_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# # ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_VIC0583_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_VIC0583_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))


# ## VIC0829
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0829/Cz_EEGvariance_VIC0829_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0829/Cz_EEGauto_VIC0829_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
#
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
#
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0829/RRI_ch31_rawvariance_VIC0829_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0829/RRI_ch31_rawauto_VIC0829_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_RRI31_arr))
# index_1=[6872,10359,16528,33444,34881,36505,38528] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((39760,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# # print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_PNES_VIC0829_features.csv")
#
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_VIC0829_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# # ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_VIC0829_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_VIC0829_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))



# ## ACT0128
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/ACT0128/Cz_EEGvariance_ACT0128_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/ACT0128/Cz_EEGauto_ACT0128_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/ACT0128/RRI_ch31_rawvariance_ACT0128_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/ACT0128/RRI_ch31_rawauto_ACT0128_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_RRI31_arr))
# index_1=[1701,6781,11290,13475,18450,23500,28619] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((35369,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
#
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# # print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_PNES_ACT0128_features.csv")
#
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_ACT0128_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# # ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_ACT0128_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_PNES_ACT0128_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))









# ### epilepsy epilepsy epilepsy epilepsy epilepsy
# ## SA0124
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEG_timewindowarr_SA0124_15s.csv',sep=',',header=None)
# t_window_arr= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGvariance_SA0124_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGauto_SA0124_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_EEG_arr))
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_timewindowarr_SA0124_15s_3h.csv',sep=',',header=None)
# rri_t= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawvariance_SA0124_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawauto_SA0124_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
#
# index_1=[1487,3829,5081,7195,10117,12498,15489,18697,21542,24461,27043,29002,29974,31715,32778] ## leading seizure index
# index_2=[1953,2050,2337,2602,3013,4033,5571,7574,7997,8359,8736,10357,10783,11012,19625,19899,21912,25494,26145,27471,27979,28105,30389,33907,36446,39066] ### seizure cluster index
# index=index_1+index_2
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((39251,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_ES_SA0124_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_SA0124_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_SA0124_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_SA0124_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
#
#
#
# ## QLD0098
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGvariance_QLD0098_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGauto_QLD0098_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_EEG_arr))
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawvariance_QLD0098_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawauto_QLD0098_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# index_1=[6793,12505,18409,25580,29963,30373,35833] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((38160,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_ES_QLD0098_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_QLD0098_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_QLD0098_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_QLD0098_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
#
#
#
# ## QLD0227
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGvariance_QLD0227_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGauto_QLD0227_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_EEG_arr))
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawvariance_QLD0227_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawauto_QLD0227_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
#
# index_1=[10228,14828,15269,17045,18749,26832] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((39280,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_ES_QLD0227_features.csv")
#
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_QLD0227_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_QLD0227_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_QLD0227_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
#
#
#
# ## VIC1202
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.Oct/Cz_EEGvariance_VIC1202_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.Oct/Cz_EEGauto_VIC1202_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_EEG_arr))
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1202/RRI_ch31_rawvariance_VIC1202_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1202/RRI_ch31_rawauto_VIC1202_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# index_1=[5168,8954,10700,11824,18724,20645,31522,32883,34118,36264,37838,39684,41631] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((54610,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1202_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1202_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1202_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1202_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
#
#
#
#
# ## VIC1173
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.Oct/Cz_EEGvariance_VIC1173_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.Oct/Cz_EEGauto_VIC1173_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_EEG_arr))
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1173/RRI_ch31_rawvariance_VIC1173_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1173/RRI_ch31_rawauto_VIC1173_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# index_1=[4541, 11615, 22100, 29041, 30203, 32982, 34902] ## leading seizure index
# index_2=[22803, 30340, 30403, 35364, 35492]
# index=index_1+index_2
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((44636,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1173_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1173_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1173_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1173_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))


# ## VIC1757
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1757/Cz_EEGvariance_VIC1757_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1757/Cz_EEGvariance_VIC1757_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_EEG_arr))
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1757/RRI_ch31_rawvariance_VIC1757_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1757/RRI_ch31_rawauto_VIC1757_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# index_1=[1482, 4069, 6275, 7236, 8748, 10134, 11460,12876,19165,21696,24439,26535] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((55262,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1757_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1757_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1757_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1757_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
#
#
# ## VIC2284
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC2284/Cz_EEGvariance_VIC2284_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC2284/Cz_EEGvariance_VIC2284_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_EEG_arr))
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC2284/RRI_ch31_rawvariance_VIC2284_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC2284/RRI_ch31_rawauto_VIC2284_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# index_1=[3143, 5359, 8943, 14823, 21543, 22503, 26619,32395] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((36350,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC2284_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC2284_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC2284_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC2284_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
#
#
#
# ## QLD0481
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/QLD0481/Cz_EEGvariance_QLD0481_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/QLD0481/Cz_EEGvariance_QLD0481_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_EEG_arr))
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/QLD0481/RRI_ch31_rawvariance_QLD0481_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/QLD0481/RRI_ch31_rawauto_QLD0481_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# index_1=[4976, 15642, 17396, 21651, 26941, 32448, 33810,38639,44069,45230,49534] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((60764,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_ES_QLD0481_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_QLD0481_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_QLD0481_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_QLD0481_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
#
#
#
# ## VIC1795
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1795/Cz_EEGvariance_VIC1795_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1795/Cz_EEGvariance_VIC1795_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_EEG_arr))
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1795/RRI_ch31_rawvariance_VIC1795_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1795/RRI_ch31_rawauto_VIC1795_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# index_1=[4449, 5709, 10241, 11369, 16425, 17297] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((37352,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1795_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1795_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1795_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1795_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
#
#
#
# # NSW0352
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/NSW0352/Cz_EEGvariance_NSW0352_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/NSW0352/Cz_EEGvariance_NSW0352_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_EEG_arr))
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/NSW0352/RRI_ch31_rawvariance_NSW0352_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/NSW0352/RRI_ch31_rawauto_NSW0352_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# index_1=[3990, 4857, 6571, 15626, 18008, 21199, 28989,29737,35520] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((39485,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_ES_NSW0352_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_NSW0352_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_NSW0352_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_NSW0352_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))



# # VIC0251
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGvariance_VIC0251_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGauto_VIC0251_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_EEG_arr))
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_rawvariance_VIC0251_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_rawauto_VIC0251_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# index_1=[1958, 6712, 9530, 12549, 13636, 18130, 20847,22055,22889,26682,28297,32575] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((39196,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC0251_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC0251_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC0251_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC0251_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
#
#
# # SA0174
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/SA0174/Cz_EEGvariance_SA0174_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/SA0174/Cz_EEGauto_SA0174_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_EEG_arr))
#
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/SA0174/RRI_ch31_rawvariance_SA0174_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/SA0174/RRI_ch31_rawauto_SA0174_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# index_1=[1440, 4840, 7397, 10079, 12304, 14471, 19460,27346,31581,35983,37739] ## leading seizure index
# index=index_1
#
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((39884,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_ES_SA0174_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_SA0174_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_SA0174_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_SA0174_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
#
#
#
#
# # VIC1027
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1027/Cz_EEGvariance_VIC1027_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1027/Cz_EEGauto_VIC1027_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_EEG_arr))
#
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1027/RRI_ch31_rawvariance_VIC1027_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1027/RRI_ch31_rawauto_VIC1027_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
#
# index_1=[3425, 10184, 11694, 15206, 17250, 20132] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
#
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((33395,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
#
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1027_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1027_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1027_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1027_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
#
#
#
# # VIC0685
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC0685/Cz_EEGvariance_VIC0685_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC0685/Cz_EEGauto_VIC0685_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_EEG_arr))
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC0685/RRI_ch31_rawvariance_VIC0685_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC0685/RRI_ch31_rawauto_VIC0685_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# index_1=[6523, 10498, 13594, 15604, 27593, 33974] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((38895,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC0685_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC0685_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC0685_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC0685_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
#
#
#
# # TAS0102
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/TAS0102/Cz_EEGvariance_TAS0102_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/TAS0102/Cz_EEGauto_TAS0102_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_EEG_arr))
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/TAS0102/RRI_ch31_rawvariance_TAS0102_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/TAS0102/RRI_ch31_rawauto_TAS0102_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# index_1=[12354, 13742, 18039, 19526, 27936, 34984] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((38617,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_ES_TAS0102_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_TAS0102_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_TAS0102_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_TAS0102_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
#
#
#
# # VIC1006
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1006/Cz_EEGvariance_VIC1006_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1006/Cz_EEGauto_VIC1006_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
# print(len(Raw_auto_EEG_arr))
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1006/RRI_ch31_rawvariance_VIC1006_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1006/RRI_ch31_rawauto_VIC1006_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
# index_1=[5549, 6863, 10707, 12202, 17890, 22206,24422,27912,29403,34144,35421] ## leading seizure index
# index=index_1
# seizure_index=[]
# for k in index:
#     seizure_index=seizure_index+list(np.linspace(k-240, k, num=240).astype(int))
# print(seizure_index);print(len(seizure_index))
#
# nonseizure_index_array=[]
# for k in index:
#     nonseizure_index_array.append(k-960)
# nonseizure_index=[]
# for item in nonseizure_index_array:
#     nonseizure_index=nonseizure_index+list(np.linspace(item-240, item, num=240).astype(int))
# print(nonseizure_index);print(len(nonseizure_index))
# seizure_index_list=[]
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
#         nonseizure_index.remove(n)
#     else:
#         seizure_index_list.append(n)
# print(a)
# seizure_index=seizure_index_list
# print(len(seizure_index));print(len(nonseizure_index));
# a=0
# for n in seizure_index:
#     if n in nonseizure_index:
#         a=a+1
# print(a)
# class_arr=np.zeros((39418,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,2*len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data.loc[wt[2*i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[seizure_index[i]]
#     data.loc[wt[2*i+1], 'EEGvar':'Class'] = Raw_variance_EEG_arr[nonseizure_index[i]], Raw_auto_EEG_arr[nonseizure_index[i]],Raw_variance_RRI31_arr[nonseizure_index[i]],Raw_auto_RRI31_arr[nonseizure_index[i]],Raw_class_arr[nonseizure_index[i]]
# print(data.shape);print(data);
# pd.DataFrame(data).to_csv("C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1006_features.csv")
# dataset=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1006_features.csv',sep=',',usecols=[1,2,3,4,5])
# ### summarise dataset
# print(dataset.shape); print(dataset.head(20));print(dataset.describe());print(dataset.groupby('Class').size());
# ### data visualization
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1006_features.csv',sep=',',usecols=[1,2,3,4])
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# data.hist()
# pyplot.show()
# scatter_matrix(data)
# pyplot.show()
# data=pd.read_csv('C:/Users/wxiong/Documents/PHD/result/ML_ES_VIC1006_features.csv',sep=',',usecols=[1,2,3,4,5])
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
# print(X_train);print(X_test);print(y_train);print(y_test);
# ### Compare Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### SVM classifier
# clf=svm.SVC(gamma='auto')
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))

