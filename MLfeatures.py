from __future__ import division
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter,iirfilter,filtfilt



def movingaverage(values, window_size):
    weights = (np.ones(window_size))/window_size
    a=np.ones(1)
    return lfilter(weights,a,values)


## TAS0056
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/TAS0056/Cz_EEGvariance_TAS0056_15s_3h.csv',sep=',',header=None)
Raw_variance_EEG= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/TAS0056/Cz_EEGauto_TAS0056_15s_3h.csv',sep=',',header=None)
Raw_auto_EEG= csv_reader.values
Raw_variance_EEG_arr=[]
for item in Raw_variance_EEG:
    Raw_variance_EEG_arr.append(float(item))
Raw_auto_EEG_arr=[]
for item in Raw_auto_EEG:
    Raw_auto_EEG_arr.append(float(item))

medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
Raw_variance_EEG_arr=medium_rhythm_var_arr_3
Raw_auto_EEG_arr=medium_rhythm_value_arr_3

csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/TAS0056/RRI_ch31_rawvariance_TAS0056_15s_3h.csv',sep=',',header=None)
RRI_var= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/TAS0056/RRI_ch31_rawauto_TAS0056_15s_3h.csv',sep=',',header=None)
Raw_auto_RRI31= csv_reader.values
Raw_variance_RRI31_arr=[]
for item in RRI_var:
    Raw_variance_RRI31_arr.append(float(item))
Raw_auto_RRI31_arr=[]
for item in Raw_auto_RRI31:
    Raw_auto_RRI31_arr.append(float(item))
medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
Raw_auto_RRI31_arr=medium_rhythm_value_arr_3

seizure_index_1=np.linspace(1475-240, 1475, num=240).astype(int)
seizure_index_2=np.linspace(2541-240, 2541, num=240).astype(int)
seizure_index_3=np.linspace(4234-240, 4234, num=240).astype(int)
seizure_index_4=np.linspace(8370-240, 8370, num=240).astype(int)
seizure_index_5=np.linspace(10353-240, 10353, num=240).astype(int)
seizure_index_6=np.linspace(11647-240, 11647, num=240).astype(int)
seizure_index_7=np.linspace(14479-240, 14479, num=240).astype(int)
seizure_index_8=np.linspace(18809-240, 18809, num=240).astype(int)
seizure_index_9=np.linspace(20161-240, 20161, num=240).astype(int)
seizure_index_10=np.linspace(21596-240, 21596, num=240).astype(int)
seizure_index_11=np.linspace(25736-240, 25736, num=240).astype(int)
seizure_index_12=np.linspace(26497-240, 26497, num=240).astype(int)
seizure_index_13=np.linspace(27232-240, 27232, num=240).astype(int)
seizure_index_14=np.linspace(31995-240, 31995, num=240).astype(int)
seizure_index_15=np.linspace(37691-240, 37691, num=240).astype(int)
seizure_index_16=np.linspace(39155-240, 39155, num=240).astype(int)

seizure_index_17=np.linspace(8748-240, 8748, num=240).astype(int)
seizure_index_18=np.linspace(9074-240, 9074, num=240).astype(int)
seizure_index_19=np.linspace(9619-240, 9619, num=240).astype(int)
seizure_index_20=np.linspace(14600-240, 14600, num=240).astype(int)
seizure_index_21=np.linspace(14863-240, 14863, num=240).astype(int)
seizure_index_22=np.linspace(14926-240, 14926, num=240).astype(int)
seizure_index_23=np.linspace(15453-240, 15453, num=240).astype(int)
seizure_index_24=np.linspace(27562-240, 27562, num=240).astype(int)
seizure_index_25=np.linspace(27754-240, 27754, num=240).astype(int)
seizure_index_26=np.linspace(32269-240, 32269, num=240).astype(int)
seizure_index_27=np.linspace(38075-240, 38075, num=240).astype(int)
seizure_index_28=np.linspace(38423-240, 38423, num=240).astype(int)

seizure_index=list(seizure_index_1)+list(seizure_index_2)+list(seizure_index_3)+list(seizure_index_4)+list(seizure_index_5)+list(seizure_index_6)+list(seizure_index_7)+list(seizure_index_8)+list(seizure_index_9)+list(seizure_index_10)+list(seizure_index_11)+list(seizure_index_12)+list(seizure_index_13)+list(seizure_index_14)+list(seizure_index_15)+list(seizure_index_16)+list(seizure_index_17)+list(seizure_index_18)+list(seizure_index_19)+list(seizure_index_20)+list(seizure_index_21)+list(seizure_index_22)+list(seizure_index_23)+list(seizure_index_24)+list(seizure_index_25)+list(seizure_index_26)+list(seizure_index_27)+list(seizure_index_28)
print(seizure_index);print(len(seizure_index))

class_arr=np.zeros((39845,1))
for idx in seizure_index:
    class_arr[idx]=1
Raw_class_arr=[]
for item in class_arr:
    Raw_class_arr.append(float(item))


wt=[str(i)for i in range(0,39845)]
names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
data=pd.DataFrame(columns=[*names],index=wt)
for i in range(39845):
    data.loc[wt[i],'EEGvar':'Class']=Raw_variance_EEG_arr[i],Raw_auto_EEG_arr[i],Raw_variance_RRI31_arr[i],Raw_auto_RRI31_arr[i],Raw_class_arr[i]
print(data);print(data.shape)


# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
#
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# print(X_train);print(X_test);
#
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
#
# ### random forest classifier
# rfc=RandomForestClassifier(n_estimators=200)
# rfc.fit(X_train,y_train)
# pred_rfc=rfc.predict(X_test)
# print(classification_report(y_test,pred_rfc))
# print(confusion_matrix(y_test,pred_rfc))
# ### SVM classifier
# clf=svm.SVC()
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
# #### neural network
# mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=200)
# mlpc.fit(X_train,y_train)
# pred_mlpc=mlpc.predict(X_test)
# print(classification_report(y_test,pred_mlpc))
# print(confusion_matrix(y_test,pred_mlpc))




# wt=[str(i)for i in range(0,39845)]
# names=['EEGvar','EEGauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(39845):
#     data.loc[wt[i],'EEGvar':'Class']=Raw_variance_EEG_arr[i],Raw_auto_EEG_arr[i],Raw_class_arr[i]
# print(data);print(data.shape)
# # data.to_csv('features.csv', index=False)
#
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
#
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# print(X_train);print(X_test);
#
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
#
# ### random forest classifier
# rfc=RandomForestClassifier(n_estimators=200)
# rfc.fit(X_train,y_train)
# pred_rfc=rfc.predict(X_test)
# # print(pred_rfc)
# ## letls see how our model performed
# print(classification_report(y_test,pred_rfc))
# print(confusion_matrix(y_test,pred_rfc))
# ### SVM classifier
# clf=svm.SVC()
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
# #### neural network
# mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=200)
# mlpc.fit(X_train,y_train)
# pred_mlpc=mlpc.predict(X_test)
# print(classification_report(y_test,pred_mlpc))
# print(confusion_matrix(y_test,pred_mlpc))
#
#
# wt=[str(i)for i in range(0,39845)]
# names=['RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(39845):
#     data.loc[wt[i],'RRIvar':'Class']=Raw_variance_RRI31_arr[i],Raw_auto_RRI31_arr[i],Raw_class_arr[i]
# print(data);print(data.shape)
# # data.to_csv('features.csv', index=False)
#
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
#
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# print(X_train);print(X_test);
#
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
#
# ### random forest classifier
# rfc=RandomForestClassifier(n_estimators=200)
# rfc.fit(X_train,y_train)
# pred_rfc=rfc.predict(X_test)
# # print(pred_rfc)
# ## letls see how our model performed
# print(classification_report(y_test,pred_rfc))
# print(confusion_matrix(y_test,pred_rfc))
# ### SVM classifier
# clf=svm.SVC()
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
# #### neural network
# mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=200)
# mlpc.fit(X_train,y_train)
# pred_mlpc=mlpc.predict(X_test)
# print(classification_report(y_test,pred_mlpc))
# print(confusion_matrix(y_test,pred_mlpc))







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
#
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
#
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
#
#
# seizure_index_1=np.linspace(1487-240, 1487, num=240).astype(int)
# seizure_index_2=np.linspace(3829-240, 3829, num=240).astype(int)
# seizure_index_3=np.linspace(5081-240, 5081, num=240).astype(int)
# seizure_index_4=np.linspace(7195-240, 7195, num=240).astype(int)
# seizure_index_5=np.linspace(10117-240, 10117, num=240).astype(int)
# seizure_index_6=np.linspace(12498-240, 12498, num=240).astype(int)
# seizure_index_7=np.linspace(15489-240, 15489, num=240).astype(int)
# seizure_index_8=np.linspace(18697-240, 18697, num=240).astype(int)
# seizure_index_9=np.linspace(21542-240, 21542, num=240).astype(int)
# seizure_index_10=np.linspace(24461-240, 24461, num=240).astype(int)
# seizure_index_11=np.linspace(27043-240, 27043, num=240).astype(int)
# seizure_index_12=np.linspace(29002-240, 29002, num=240).astype(int)
# seizure_index_13=np.linspace(29974-240, 29974, num=240).astype(int)
# seizure_index_14=np.linspace(31715-240, 31715, num=240).astype(int)
# seizure_index_15=np.linspace(32778-240, 32778, num=240).astype(int)
#
# seizure_index_16=np.linspace(1953-240, 1953, num=240).astype(int)
# seizure_index_17=np.linspace(2050-240, 2050, num=240).astype(int)
# seizure_index_18=np.linspace(2337-240, 2337, num=240).astype(int)
# seizure_index_19=np.linspace(2602-240, 2602, num=240).astype(int)
# seizure_index_20=np.linspace(3013-240, 3013, num=240).astype(int)
# seizure_index_21=np.linspace(4033-240, 4033, num=240).astype(int)
# seizure_index_22=np.linspace(5571-240, 5571, num=240).astype(int)
# seizure_index_23=np.linspace(7574-240, 7574, num=240).astype(int)
# seizure_index_24=np.linspace(7997-240, 7997, num=240).astype(int)
# seizure_index_25=np.linspace(8359-240, 8359, num=240).astype(int)
# seizure_index_26=np.linspace(8736-240, 8736, num=240).astype(int)
# seizure_index_27=np.linspace(10357-240, 10357, num=240).astype(int)
# seizure_index_28=np.linspace(10783-240, 10783, num=240).astype(int)
# seizure_index_29=np.linspace(11012-240, 11012, num=240).astype(int)
# seizure_index_30=np.linspace(19625-240, 19625, num=240).astype(int)
# seizure_index_31=np.linspace(19899-240, 19899, num=240).astype(int)
# seizure_index_32=np.linspace(21912-240, 21912, num=240).astype(int)
# seizure_index_33=np.linspace(25494-240, 25494, num=240).astype(int)
# seizure_index_34=np.linspace(26145-240, 26145, num=240).astype(int)
# seizure_index_35=np.linspace(27471-240, 27471, num=240).astype(int)
# seizure_index_36=np.linspace(27979-240, 27979, num=240).astype(int)
# seizure_index_37=np.linspace(28105-240, 28105, num=240).astype(int)
# seizure_index_38=np.linspace(30389-240, 30389, num=240).astype(int)
# seizure_index_39=np.linspace(33907-240*4, 33907, num=240*4).astype(int)
# seizure_index_40=np.linspace(36446-240, 36446, num=240).astype(int)
# seizure_index_41=np.linspace(39066-240, 39066, num=240).astype(int)
#
#
#
#
# seizure_index=list(seizure_index_1)+list(seizure_index_2)+list(seizure_index_3)+list(seizure_index_4)+list(seizure_index_5)+list(seizure_index_6)+list(seizure_index_7)+list(seizure_index_8)+list(seizure_index_9)+list(seizure_index_10)+list(seizure_index_11)+list(seizure_index_12)+list(seizure_index_13)+list(seizure_index_14)+list(seizure_index_15)+list(seizure_index_16)+list(seizure_index_17)+list(seizure_index_18)+list(seizure_index_19)+list(seizure_index_20)+list(seizure_index_21)+list(seizure_index_22)+list(seizure_index_23)+list(seizure_index_24)+list(seizure_index_25)+list(seizure_index_26)+list(seizure_index_27)+list(seizure_index_28)+list(seizure_index_29)+list(seizure_index_30)+list(seizure_index_31)+list(seizure_index_32)+list(seizure_index_33)+list(seizure_index_34)+list(seizure_index_35)+list(seizure_index_36)+list(seizure_index_37)+list(seizure_index_38)+list(seizure_index_39)+list(seizure_index_40)+list(seizure_index_41)
# print(seizure_index)

# class_arr=np.zeros((39845,1))
# for idx in seizure_index:
#     class_arr[idx]=1
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))

# wt=[str(i)for i in range(0,39251)]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(39251):
#     data.loc[wt[i],'EEGvar':'Class']=Raw_variance_EEG_arr[i],Raw_auto_EEG_arr[i],Raw_variance_RRI31_arr[i],Raw_auto_RRI31_arr[i],Raw_class_arr[i]
# print(data);print(data.shape)
# # data.to_csv('features.csv', index=False)
#
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
#
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# print(X_train);print(X_test);
#
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
#
# # ### random forest classifier
# # rfc=RandomForestClassifier(n_estimators=200)
# # rfc.fit(X_train,y_train)
# # pred_rfc=rfc.predict(X_test)
# # # print(pred_rfc)
# # ## letls see how our model performed
# # print(classification_report(y_test,pred_rfc))
# # print(confusion_matrix(y_test,pred_rfc))
#
# # ### SVM classifier
# # clf=svm.SVC()
# # clf.fit(X_train,y_train)
# # pred_clf=clf.predict(X_test)
# # print(classification_report(y_test,pred_clf))
# # print(confusion_matrix(y_test,pred_clf))
# #
# # #### neural network
# # mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=200)
# # mlpc.fit(X_train,y_train)
# # pred_mlpc=mlpc.predict(X_test)
# # print(classification_report(y_test,pred_mlpc))
# # print(confusion_matrix(y_test,pred_mlpc))
#
#
# wt=[str(i)for i in range(0,39251)]
# names=['EEGvar','EEGauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(39251):
#     data.loc[wt[i],'EEGvar':'Class']=Raw_variance_EEG_arr[i],Raw_auto_EEG_arr[i],Raw_class_arr[i]
# print(data);print(data.shape)
# # data.to_csv('features.csv', index=False)
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# print(X_train);print(X_test);
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### random forest classifier
# rfc=RandomForestClassifier(n_estimators=200)
# rfc.fit(X_train,y_train)
# pred_rfc=rfc.predict(X_test)
# # print(pred_rfc)
# ## letls see how our model performed
# print(classification_report(y_test,pred_rfc))
# print(confusion_matrix(y_test,pred_rfc))
# ### SVM classifier
# clf=svm.SVC()
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
# #### neural network
# mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=200)
# mlpc.fit(X_train,y_train)
# pred_mlpc=mlpc.predict(X_test)
# print(classification_report(y_test,pred_mlpc))
# print(confusion_matrix(y_test,pred_mlpc))
#
#
#
#
#
#
# wt=[str(i)for i in range(0,39251)]
# names=['RRIvar','RRIauto','Class']
# data=pd.DataFrame(columns=[*names],index=wt)
# for i in range(39251):
#     data.loc[wt[i],'RRIvar':'Class']=Raw_variance_RRI31_arr[i],Raw_auto_RRI31_arr[i],Raw_class_arr[i]
# print(data);print(data.shape)
# # data.to_csv('features.csv', index=False)
# X=data.drop('Class',axis=1)
# y=data['Class']
# print(X);print(y)
# y=y.astype('int')
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# print(X_train);print(X_test);
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### random forest classifier
# rfc=RandomForestClassifier(n_estimators=200)
# rfc.fit(X_train,y_train)
# pred_rfc=rfc.predict(X_test)
# # print(pred_rfc)
# ## letls see how our model performed
# print(classification_report(y_test,pred_rfc))
# print(confusion_matrix(y_test,pred_rfc))
# ### SVM classifier
# clf=svm.SVC()
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
# #### neural network
# mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=200)
# mlpc.fit(X_train,y_train)
# pred_mlpc=mlpc.predict(X_test)
# print(classification_report(y_test,pred_mlpc))
# print(confusion_matrix(y_test,pred_mlpc))







# ### epilepsy and PNES
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
#
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
#
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
#
# seizure_index_1=np.linspace(1475-240, 1475, num=240).astype(int)
# seizure_index_2=np.linspace(2541-240, 2541, num=240).astype(int)
# seizure_index_3=np.linspace(4234-240, 4234, num=240).astype(int)
# seizure_index_4=np.linspace(8370-240, 8370, num=240).astype(int)
# seizure_index_5=np.linspace(10353-240, 10353, num=240).astype(int)
# seizure_index_6=np.linspace(11647-240, 11647, num=240).astype(int)
# seizure_index_7=np.linspace(14479-240, 14479, num=240).astype(int)
# seizure_index_8=np.linspace(18809-240, 18809, num=240).astype(int)
# seizure_index_9=np.linspace(20161-240, 20161, num=240).astype(int)
# seizure_index_10=np.linspace(21596-240, 21596, num=240).astype(int)
# seizure_index_11=np.linspace(25736-240, 25736, num=240).astype(int)
# seizure_index_12=np.linspace(26497-240, 26497, num=240).astype(int)
# seizure_index_13=np.linspace(27232-240, 27232, num=240).astype(int)
# seizure_index_14=np.linspace(31995-240, 31995, num=240).astype(int)
# seizure_index_15=np.linspace(37691-240, 37691, num=240).astype(int)
# seizure_index_16=np.linspace(39155-240, 39155, num=240).astype(int)
#
# seizure_index_17=np.linspace(8748-240, 8748, num=240).astype(int)
# seizure_index_18=np.linspace(9074-240, 9074, num=240).astype(int)
# seizure_index_19=np.linspace(9619-240, 9619, num=240).astype(int)
# seizure_index_20=np.linspace(14600-240, 14600, num=240).astype(int)
# seizure_index_21=np.linspace(14863-240, 14863, num=240).astype(int)
# seizure_index_22=np.linspace(14926-240, 14926, num=240).astype(int)
# seizure_index_23=np.linspace(15453-240, 15453, num=240).astype(int)
# seizure_index_24=np.linspace(27562-240, 27562, num=240).astype(int)
# seizure_index_25=np.linspace(27754-240, 27754, num=240).astype(int)
# seizure_index_26=np.linspace(32269-240, 32269, num=240).astype(int)
# seizure_index_27=np.linspace(38075-240, 38075, num=240).astype(int)
# seizure_index_28=np.linspace(38423-240, 38423, num=240).astype(int)
#
# seizure_index=list(seizure_index_1)+list(seizure_index_2)+list(seizure_index_3)+list(seizure_index_4)+list(seizure_index_5)+list(seizure_index_6)+list(seizure_index_7)+list(seizure_index_8)+list(seizure_index_9)+list(seizure_index_10)+list(seizure_index_11)+list(seizure_index_12)+list(seizure_index_13)+list(seizure_index_14)+list(seizure_index_15)+list(seizure_index_16)+list(seizure_index_17)+list(seizure_index_18)+list(seizure_index_19)+list(seizure_index_20)+list(seizure_index_21)+list(seizure_index_22)+list(seizure_index_23)+list(seizure_index_24)+list(seizure_index_25)+list(seizure_index_26)+list(seizure_index_27)
# print(seizure_index);print(len(seizure_index))
#
# class_arr=np.zeros((len(seizure_index),1))
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data1=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data1.loc[wt[i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[i]
# print(data1);print(data1.shape)
#
#
#
#
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
#
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
#
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
#
#
# seizure_index_1=np.linspace(1487-240, 1487, num=240).astype(int)
# seizure_index_2=np.linspace(3829-240, 3829, num=240).astype(int)
# seizure_index_3=np.linspace(5081-240, 5081, num=240).astype(int)
# seizure_index_4=np.linspace(7195-240, 7195, num=240).astype(int)
# seizure_index_5=np.linspace(10117-240, 10117, num=240).astype(int)
# seizure_index_6=np.linspace(12498-240, 12498, num=240).astype(int)
# seizure_index_7=np.linspace(15489-240, 15489, num=240).astype(int)
# seizure_index_8=np.linspace(18697-240, 18697, num=240).astype(int)
# seizure_index_9=np.linspace(21542-240, 21542, num=240).astype(int)
# seizure_index_10=np.linspace(24461-240, 24461, num=240).astype(int)
# seizure_index_11=np.linspace(27043-240, 27043, num=240).astype(int)
# seizure_index_12=np.linspace(29002-240, 29002, num=240).astype(int)
# seizure_index_13=np.linspace(29974-240, 29974, num=240).astype(int)
# seizure_index_14=np.linspace(31715-240, 31715, num=240).astype(int)
# seizure_index_15=np.linspace(32778-240, 32778, num=240).astype(int)
#
# seizure_index_16=np.linspace(1953-240, 1953, num=240).astype(int)
# seizure_index_17=np.linspace(2050-240, 2050, num=240).astype(int)
# seizure_index_18=np.linspace(2337-240, 2337, num=240).astype(int)
# seizure_index_19=np.linspace(2602-240, 2602, num=240).astype(int)
# seizure_index_20=np.linspace(3013-240, 3013, num=240).astype(int)
# seizure_index_21=np.linspace(4033-240, 4033, num=240).astype(int)
# seizure_index_22=np.linspace(5571-240, 5571, num=240).astype(int)
# seizure_index_23=np.linspace(7574-240, 7574, num=240).astype(int)
# seizure_index_24=np.linspace(7997-240, 7997, num=240).astype(int)
# seizure_index_25=np.linspace(8359-240, 8359, num=240).astype(int)
# seizure_index_26=np.linspace(8736-240, 8736, num=240).astype(int)
# seizure_index_27=np.linspace(10357-240, 10357, num=240).astype(int)
# seizure_index_28=np.linspace(10783-240, 10783, num=240).astype(int)
# seizure_index_29=np.linspace(11012-240, 11012, num=240).astype(int)
# seizure_index_30=np.linspace(19625-240, 19625, num=240).astype(int)
# seizure_index_31=np.linspace(19899-240, 19899, num=240).astype(int)
# seizure_index_32=np.linspace(21912-240, 21912, num=240).astype(int)
# seizure_index_33=np.linspace(25494-240, 25494, num=240).astype(int)
# seizure_index_34=np.linspace(26145-240, 26145, num=240).astype(int)
# seizure_index_35=np.linspace(27471-240, 27471, num=240).astype(int)
# seizure_index_36=np.linspace(27979-240, 27979, num=240).astype(int)
# seizure_index_37=np.linspace(28105-240, 28105, num=240).astype(int)
# seizure_index_38=np.linspace(30389-240, 30389, num=240).astype(int)
# seizure_index_39=np.linspace(33907-240*4, 33907, num=240*4).astype(int)
# seizure_index_40=np.linspace(36446-240, 36446, num=240).astype(int)
# seizure_index_41=np.linspace(39066-240, 39066, num=240).astype(int)
#
# seizure_index=list(seizure_index_1)+list(seizure_index_2)+list(seizure_index_3)+list(seizure_index_4)+list(seizure_index_5)+list(seizure_index_6)+list(seizure_index_7)+list(seizure_index_8)+list(seizure_index_9)+list(seizure_index_10)+list(seizure_index_11)+list(seizure_index_12)+list(seizure_index_13)+list(seizure_index_14)+list(seizure_index_15)+list(seizure_index_16)+list(seizure_index_17)+list(seizure_index_18)+list(seizure_index_19)+list(seizure_index_20)+list(seizure_index_21)+list(seizure_index_22)+list(seizure_index_23)+list(seizure_index_24)+list(seizure_index_25)+list(seizure_index_26)+list(seizure_index_27)+list(seizure_index_28)+list(seizure_index_29)+list(seizure_index_30)+list(seizure_index_31)+list(seizure_index_32)+list(seizure_index_33)+list(seizure_index_34)+list(seizure_index_35)+list(seizure_index_36)+list(seizure_index_37)+list(seizure_index_38)+list(seizure_index_39)+list(seizure_index_40)+list(seizure_index_41)
# print(seizure_index)
#
# class_arr=np.ones((len(seizure_index),1))
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,len(seizure_index))]
# names=['EEGvar','EEGauto','RRIvar','RRIauto','Class']
# data2=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data2.loc[wt[i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[i]
# print(data2);print(data2.shape)
# data=[data1,data2]
# result = pd.concat(data,ignore_index=True)
# print(result)
#
# X=result.drop('Class',axis=1)
# y=result['Class']
# print(X);print(y)
# y=y.astype('int')
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# print(X_train);print(X_test);
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### random forest classifier
# rfc=RandomForestClassifier(n_estimators=200)
# rfc.fit(X_train,y_train)
# pred_rfc=rfc.predict(X_test)
# # print(pred_rfc)
# ## letls see how our model performed
# print(classification_report(y_test,pred_rfc))
# print(confusion_matrix(y_test,pred_rfc))
# ### SVM classifier
# clf=svm.SVC()
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
# #### neural network
# mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=200)
# mlpc.fit(X_train,y_train)
# pred_mlpc=mlpc.predict(X_test)
# print(classification_report(y_test,pred_mlpc))
# print(confusion_matrix(y_test,pred_mlpc))





# ### epilepsy and PNES
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
#
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
# Raw_variance_EEG_arr=medium_rhythm_var_arr_3
# Raw_auto_EEG_arr=medium_rhythm_value_arr_3
#
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
#
# seizure_index_1=np.linspace(1475-240, 1475, num=240).astype(int)
# seizure_index_2=np.linspace(2541-240, 2541, num=240).astype(int)
# seizure_index_3=np.linspace(4234-240, 4234, num=240).astype(int)
# seizure_index_4=np.linspace(8370-240, 8370, num=240).astype(int)
# seizure_index_5=np.linspace(10353-240, 10353, num=240).astype(int)
# seizure_index_6=np.linspace(11647-240, 11647, num=240).astype(int)
# seizure_index_7=np.linspace(14479-240, 14479, num=240).astype(int)
# seizure_index_8=np.linspace(18809-240, 18809, num=240).astype(int)
# seizure_index_9=np.linspace(20161-240, 20161, num=240).astype(int)
# seizure_index_10=np.linspace(21596-240, 21596, num=240).astype(int)
# seizure_index_11=np.linspace(25736-240, 25736, num=240).astype(int)
# seizure_index_12=np.linspace(26497-240, 26497, num=240).astype(int)
# seizure_index_13=np.linspace(27232-240, 27232, num=240).astype(int)
# seizure_index_14=np.linspace(31995-240, 31995, num=240).astype(int)
# seizure_index_15=np.linspace(37691-240, 37691, num=240).astype(int)
# seizure_index_16=np.linspace(39155-240, 39155, num=240).astype(int)
#
# seizure_index_17=np.linspace(8748-240, 8748, num=240).astype(int)
# seizure_index_18=np.linspace(9074-240, 9074, num=240).astype(int)
# seizure_index_19=np.linspace(9619-240, 9619, num=240).astype(int)
# seizure_index_20=np.linspace(14600-240, 14600, num=240).astype(int)
# seizure_index_21=np.linspace(14863-240, 14863, num=240).astype(int)
# seizure_index_22=np.linspace(14926-240, 14926, num=240).astype(int)
# seizure_index_23=np.linspace(15453-240, 15453, num=240).astype(int)
# seizure_index_24=np.linspace(27562-240, 27562, num=240).astype(int)
# seizure_index_25=np.linspace(27754-240, 27754, num=240).astype(int)
# seizure_index_26=np.linspace(32269-240, 32269, num=240).astype(int)
# seizure_index_27=np.linspace(38075-240, 38075, num=240).astype(int)
# seizure_index_28=np.linspace(38423-240, 38423, num=240).astype(int)
#
# seizure_index=list(seizure_index_1)+list(seizure_index_2)+list(seizure_index_3)+list(seizure_index_4)+list(seizure_index_5)+list(seizure_index_6)+list(seizure_index_7)+list(seizure_index_8)+list(seizure_index_9)+list(seizure_index_10)+list(seizure_index_11)+list(seizure_index_12)+list(seizure_index_13)+list(seizure_index_14)+list(seizure_index_15)+list(seizure_index_16)+list(seizure_index_17)+list(seizure_index_18)+list(seizure_index_19)+list(seizure_index_20)+list(seizure_index_21)+list(seizure_index_22)+list(seizure_index_23)+list(seizure_index_24)+list(seizure_index_25)+list(seizure_index_26)+list(seizure_index_27)
# print(seizure_index);print(len(seizure_index))
#
# class_arr=np.zeros((len(seizure_index),1))
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,len(seizure_index))]
# names=['EEGvar','EEGauto','Class']
# data1=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data1.loc[wt[i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_class_arr[i]
# print(data1);print(data1.shape)
#
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
#
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
#
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
# Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
# Raw_auto_RRI31_arr=medium_rhythm_value_arr_3
#
#
# seizure_index_1=np.linspace(1487-240, 1487, num=240).astype(int)
# seizure_index_2=np.linspace(3829-240, 3829, num=240).astype(int)
# seizure_index_3=np.linspace(5081-240, 5081, num=240).astype(int)
# seizure_index_4=np.linspace(7195-240, 7195, num=240).astype(int)
# seizure_index_5=np.linspace(10117-240, 10117, num=240).astype(int)
# seizure_index_6=np.linspace(12498-240, 12498, num=240).astype(int)
# seizure_index_7=np.linspace(15489-240, 15489, num=240).astype(int)
# seizure_index_8=np.linspace(18697-240, 18697, num=240).astype(int)
# seizure_index_9=np.linspace(21542-240, 21542, num=240).astype(int)
# seizure_index_10=np.linspace(24461-240, 24461, num=240).astype(int)
# seizure_index_11=np.linspace(27043-240, 27043, num=240).astype(int)
# seizure_index_12=np.linspace(29002-240, 29002, num=240).astype(int)
# seizure_index_13=np.linspace(29974-240, 29974, num=240).astype(int)
# seizure_index_14=np.linspace(31715-240, 31715, num=240).astype(int)
# seizure_index_15=np.linspace(32778-240, 32778, num=240).astype(int)
#
# seizure_index_16=np.linspace(1953-240, 1953, num=240).astype(int)
# seizure_index_17=np.linspace(2050-240, 2050, num=240).astype(int)
# seizure_index_18=np.linspace(2337-240, 2337, num=240).astype(int)
# seizure_index_19=np.linspace(2602-240, 2602, num=240).astype(int)
# seizure_index_20=np.linspace(3013-240, 3013, num=240).astype(int)
# seizure_index_21=np.linspace(4033-240, 4033, num=240).astype(int)
# seizure_index_22=np.linspace(5571-240, 5571, num=240).astype(int)
# seizure_index_23=np.linspace(7574-240, 7574, num=240).astype(int)
# seizure_index_24=np.linspace(7997-240, 7997, num=240).astype(int)
# seizure_index_25=np.linspace(8359-240, 8359, num=240).astype(int)
# seizure_index_26=np.linspace(8736-240, 8736, num=240).astype(int)
# seizure_index_27=np.linspace(10357-240, 10357, num=240).astype(int)
# seizure_index_28=np.linspace(10783-240, 10783, num=240).astype(int)
# seizure_index_29=np.linspace(11012-240, 11012, num=240).astype(int)
# seizure_index_30=np.linspace(19625-240, 19625, num=240).astype(int)
# seizure_index_31=np.linspace(19899-240, 19899, num=240).astype(int)
# seizure_index_32=np.linspace(21912-240, 21912, num=240).astype(int)
# seizure_index_33=np.linspace(25494-240, 25494, num=240).astype(int)
# seizure_index_34=np.linspace(26145-240, 26145, num=240).astype(int)
# seizure_index_35=np.linspace(27471-240, 27471, num=240).astype(int)
# seizure_index_36=np.linspace(27979-240, 27979, num=240).astype(int)
# seizure_index_37=np.linspace(28105-240, 28105, num=240).astype(int)
# seizure_index_38=np.linspace(30389-240, 30389, num=240).astype(int)
# seizure_index_39=np.linspace(33907-240*4, 33907, num=240*4).astype(int)
# seizure_index_40=np.linspace(36446-240, 36446, num=240).astype(int)
# seizure_index_41=np.linspace(39066-240, 39066, num=240).astype(int)
#
# seizure_index=list(seizure_index_1)+list(seizure_index_2)+list(seizure_index_3)+list(seizure_index_4)+list(seizure_index_5)+list(seizure_index_6)+list(seizure_index_7)+list(seizure_index_8)+list(seizure_index_9)+list(seizure_index_10)+list(seizure_index_11)+list(seizure_index_12)+list(seizure_index_13)+list(seizure_index_14)+list(seizure_index_15)+list(seizure_index_16)+list(seizure_index_17)+list(seizure_index_18)+list(seizure_index_19)+list(seizure_index_20)+list(seizure_index_21)+list(seizure_index_22)+list(seizure_index_23)+list(seizure_index_24)+list(seizure_index_25)+list(seizure_index_26)+list(seizure_index_27)+list(seizure_index_28)+list(seizure_index_29)+list(seizure_index_30)+list(seizure_index_31)+list(seizure_index_32)+list(seizure_index_33)+list(seizure_index_34)+list(seizure_index_35)+list(seizure_index_36)+list(seizure_index_37)+list(seizure_index_38)+list(seizure_index_39)+list(seizure_index_40)+list(seizure_index_41)
# print(seizure_index)
#
# class_arr=np.ones((len(seizure_index),1))
# Raw_class_arr=[]
# for item in class_arr:
#     Raw_class_arr.append(float(item))
# wt=[str(i)for i in range(0,len(seizure_index))]
# names=['EEGvar','EEGauto','Class']
# data2=pd.DataFrame(columns=[*names],index=wt)
# for i in range(len(seizure_index)):
#     data2.loc[wt[i],'EEGvar':'Class']=Raw_variance_EEG_arr[seizure_index[i]],Raw_auto_EEG_arr[seizure_index[i]],Raw_class_arr[i]
# print(data2);print(data2.shape)
# data=[data1,data2]
# result = pd.concat(data,ignore_index=True)
# print(result)
#
# X=result.drop('Class',axis=1)
# y=result['Class']
# print(X);print(y)
# y=y.astype('int')
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# print(X_train);print(X_test);
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# print(X_train);print(X_test);
# ### random forest classifier
# rfc=RandomForestClassifier(n_estimators=200)
# rfc.fit(X_train,y_train)
# pred_rfc=rfc.predict(X_test)
# # print(pred_rfc)
# ## letls see how our model performed
# print(classification_report(y_test,pred_rfc))
# print(confusion_matrix(y_test,pred_rfc))
# ### SVM classifier
# clf=svm.SVC()
# clf.fit(X_train,y_train)
# pred_clf=clf.predict(X_test)
# print(classification_report(y_test,pred_clf))
# print(confusion_matrix(y_test,pred_clf))
# #### neural network
# mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=200)
# mlpc.fit(X_train,y_train)
# pred_mlpc=mlpc.predict(X_test)
# print(classification_report(y_test,pred_mlpc))
# print(confusion_matrix(y_test,pred_mlpc))



### epilepsy and PNES
## TAS0056
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/TAS0056/Cz_EEGvariance_TAS0056_15s_3h.csv',sep=',',header=None)
Raw_variance_EEG= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/TAS0056/Cz_EEGauto_TAS0056_15s_3h.csv',sep=',',header=None)
Raw_auto_EEG= csv_reader.values
Raw_variance_EEG_arr=[]
for item in Raw_variance_EEG:
    Raw_variance_EEG_arr.append(float(item))
Raw_auto_EEG_arr=[]
for item in Raw_auto_EEG:
    Raw_auto_EEG_arr.append(float(item))

medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
Raw_variance_EEG_arr=medium_rhythm_var_arr_3
Raw_auto_EEG_arr=medium_rhythm_value_arr_3

csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/TAS0056/RRI_ch31_rawvariance_TAS0056_15s_3h.csv',sep=',',header=None)
RRI_var= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/TAS0056/RRI_ch31_rawauto_TAS0056_15s_3h.csv',sep=',',header=None)
Raw_auto_RRI31= csv_reader.values
Raw_variance_RRI31_arr=[]
for item in RRI_var:
    Raw_variance_RRI31_arr.append(float(item))
Raw_auto_RRI31_arr=[]
for item in Raw_auto_RRI31:
    Raw_auto_RRI31_arr.append(float(item))
medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
Raw_auto_RRI31_arr=medium_rhythm_value_arr_3

seizure_index_1=np.linspace(1475-240, 1475, num=240).astype(int)
seizure_index_2=np.linspace(2541-240, 2541, num=240).astype(int)
seizure_index_3=np.linspace(4234-240, 4234, num=240).astype(int)
seizure_index_4=np.linspace(8370-240, 8370, num=240).astype(int)
seizure_index_5=np.linspace(10353-240, 10353, num=240).astype(int)
seizure_index_6=np.linspace(11647-240, 11647, num=240).astype(int)
seizure_index_7=np.linspace(14479-240, 14479, num=240).astype(int)
seizure_index_8=np.linspace(18809-240, 18809, num=240).astype(int)
seizure_index_9=np.linspace(20161-240, 20161, num=240).astype(int)
seizure_index_10=np.linspace(21596-240, 21596, num=240).astype(int)
seizure_index_11=np.linspace(25736-240, 25736, num=240).astype(int)
seizure_index_12=np.linspace(26497-240, 26497, num=240).astype(int)
seizure_index_13=np.linspace(27232-240, 27232, num=240).astype(int)
seizure_index_14=np.linspace(31995-240, 31995, num=240).astype(int)
seizure_index_15=np.linspace(37691-240, 37691, num=240).astype(int)
seizure_index_16=np.linspace(39155-240, 39155, num=240).astype(int)

seizure_index_17=np.linspace(8748-240, 8748, num=240).astype(int)
seizure_index_18=np.linspace(9074-240, 9074, num=240).astype(int)
seizure_index_19=np.linspace(9619-240, 9619, num=240).astype(int)
seizure_index_20=np.linspace(14600-240, 14600, num=240).astype(int)
seizure_index_21=np.linspace(14863-240, 14863, num=240).astype(int)
seizure_index_22=np.linspace(14926-240, 14926, num=240).astype(int)
seizure_index_23=np.linspace(15453-240, 15453, num=240).astype(int)
seizure_index_24=np.linspace(27562-240, 27562, num=240).astype(int)
seizure_index_25=np.linspace(27754-240, 27754, num=240).astype(int)
seizure_index_26=np.linspace(32269-240, 32269, num=240).astype(int)
seizure_index_27=np.linspace(38075-240, 38075, num=240).astype(int)
seizure_index_28=np.linspace(38423-240, 38423, num=240).astype(int)

seizure_index=list(seizure_index_1)+list(seizure_index_2)+list(seizure_index_3)+list(seizure_index_4)+list(seizure_index_5)+list(seizure_index_6)+list(seizure_index_7)+list(seizure_index_8)+list(seizure_index_9)+list(seizure_index_10)+list(seizure_index_11)+list(seizure_index_12)+list(seizure_index_13)+list(seizure_index_14)+list(seizure_index_15)+list(seizure_index_16)+list(seizure_index_17)+list(seizure_index_18)+list(seizure_index_19)+list(seizure_index_20)+list(seizure_index_21)+list(seizure_index_22)+list(seizure_index_23)+list(seizure_index_24)+list(seizure_index_25)+list(seizure_index_26)+list(seizure_index_27)
print(seizure_index);print(len(seizure_index))

class_arr=np.zeros((len(seizure_index),1))
Raw_class_arr=[]
for item in class_arr:
    Raw_class_arr.append(float(item))
wt=[str(i)for i in range(0,len(seizure_index))]
names=['RRIvar','RRIauto','Class']
data1=pd.DataFrame(columns=[*names],index=wt)
for i in range(len(seizure_index)):
    data1.loc[wt[i],'RRIvar':'Class']=Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[i]
print(data1);print(data1.shape)

## SA0124
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEG_timewindowarr_SA0124_15s.csv',sep=',',header=None)
t_window_arr= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGvariance_SA0124_15s_3h.csv',sep=',',header=None)
Raw_variance_EEG= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGauto_SA0124_15s_3h.csv',sep=',',header=None)
Raw_auto_EEG= csv_reader.values
Raw_variance_EEG_arr=[]
for item in Raw_variance_EEG:
    Raw_variance_EEG_arr.append(float(item))
Raw_auto_EEG_arr=[]
for item in Raw_auto_EEG:
    Raw_auto_EEG_arr.append(float(item))
medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG_arr,240*6)
medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG_arr,240*6)
Raw_variance_EEG_arr=medium_rhythm_var_arr_3
Raw_auto_EEG_arr=medium_rhythm_value_arr_3

csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_timewindowarr_SA0124_15s_3h.csv',sep=',',header=None)
rri_t= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawvariance_SA0124_15s_3h.csv',sep=',',header=None)
RRI_var= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawauto_SA0124_15s_3h.csv',sep=',',header=None)
Raw_auto_RRI31= csv_reader.values
Raw_variance_RRI31_arr=[]
for item in RRI_var:
    Raw_variance_RRI31_arr.append(float(item))
Raw_auto_RRI31_arr=[]
for item in Raw_auto_RRI31:
    Raw_auto_RRI31_arr.append(float(item))

medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31_arr,240*6)
medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31_arr,240*6)
Raw_variance_RRI31_arr=medium_rhythm_var_arr_3
Raw_auto_RRI31_arr=medium_rhythm_value_arr_3


seizure_index_1=np.linspace(1487-240, 1487, num=240).astype(int)
seizure_index_2=np.linspace(3829-240, 3829, num=240).astype(int)
seizure_index_3=np.linspace(5081-240, 5081, num=240).astype(int)
seizure_index_4=np.linspace(7195-240, 7195, num=240).astype(int)
seizure_index_5=np.linspace(10117-240, 10117, num=240).astype(int)
seizure_index_6=np.linspace(12498-240, 12498, num=240).astype(int)
seizure_index_7=np.linspace(15489-240, 15489, num=240).astype(int)
seizure_index_8=np.linspace(18697-240, 18697, num=240).astype(int)
seizure_index_9=np.linspace(21542-240, 21542, num=240).astype(int)
seizure_index_10=np.linspace(24461-240, 24461, num=240).astype(int)
seizure_index_11=np.linspace(27043-240, 27043, num=240).astype(int)
seizure_index_12=np.linspace(29002-240, 29002, num=240).astype(int)
seizure_index_13=np.linspace(29974-240, 29974, num=240).astype(int)
seizure_index_14=np.linspace(31715-240, 31715, num=240).astype(int)
seizure_index_15=np.linspace(32778-240, 32778, num=240).astype(int)

seizure_index_16=np.linspace(1953-240, 1953, num=240).astype(int)
seizure_index_17=np.linspace(2050-240, 2050, num=240).astype(int)
seizure_index_18=np.linspace(2337-240, 2337, num=240).astype(int)
seizure_index_19=np.linspace(2602-240, 2602, num=240).astype(int)
seizure_index_20=np.linspace(3013-240, 3013, num=240).astype(int)
seizure_index_21=np.linspace(4033-240, 4033, num=240).astype(int)
seizure_index_22=np.linspace(5571-240, 5571, num=240).astype(int)
seizure_index_23=np.linspace(7574-240, 7574, num=240).astype(int)
seizure_index_24=np.linspace(7997-240, 7997, num=240).astype(int)
seizure_index_25=np.linspace(8359-240, 8359, num=240).astype(int)
seizure_index_26=np.linspace(8736-240, 8736, num=240).astype(int)
seizure_index_27=np.linspace(10357-240, 10357, num=240).astype(int)
seizure_index_28=np.linspace(10783-240, 10783, num=240).astype(int)
seizure_index_29=np.linspace(11012-240, 11012, num=240).astype(int)
seizure_index_30=np.linspace(19625-240, 19625, num=240).astype(int)
seizure_index_31=np.linspace(19899-240, 19899, num=240).astype(int)
seizure_index_32=np.linspace(21912-240, 21912, num=240).astype(int)
seizure_index_33=np.linspace(25494-240, 25494, num=240).astype(int)
seizure_index_34=np.linspace(26145-240, 26145, num=240).astype(int)
seizure_index_35=np.linspace(27471-240, 27471, num=240).astype(int)
seizure_index_36=np.linspace(27979-240, 27979, num=240).astype(int)
seizure_index_37=np.linspace(28105-240, 28105, num=240).astype(int)
seizure_index_38=np.linspace(30389-240, 30389, num=240).astype(int)
seizure_index_39=np.linspace(33907-240*4, 33907, num=240*4).astype(int)
seizure_index_40=np.linspace(36446-240, 36446, num=240).astype(int)
seizure_index_41=np.linspace(39066-240, 39066, num=240).astype(int)

seizure_index=list(seizure_index_1)+list(seizure_index_2)+list(seizure_index_3)+list(seizure_index_4)+list(seizure_index_5)+list(seizure_index_6)+list(seizure_index_7)+list(seizure_index_8)+list(seizure_index_9)+list(seizure_index_10)+list(seizure_index_11)+list(seizure_index_12)+list(seizure_index_13)+list(seizure_index_14)+list(seizure_index_15)+list(seizure_index_16)+list(seizure_index_17)+list(seizure_index_18)+list(seizure_index_19)+list(seizure_index_20)+list(seizure_index_21)+list(seizure_index_22)+list(seizure_index_23)+list(seizure_index_24)+list(seizure_index_25)+list(seizure_index_26)+list(seizure_index_27)+list(seizure_index_28)+list(seizure_index_29)+list(seizure_index_30)+list(seizure_index_31)+list(seizure_index_32)+list(seizure_index_33)+list(seizure_index_34)+list(seizure_index_35)+list(seizure_index_36)+list(seizure_index_37)+list(seizure_index_38)+list(seizure_index_39)+list(seizure_index_40)+list(seizure_index_41)
print(seizure_index)

class_arr=np.ones((len(seizure_index),1))
Raw_class_arr=[]
for item in class_arr:
    Raw_class_arr.append(float(item))
wt=[str(i)for i in range(0,len(seizure_index))]
names=['RRIvar','RRIauto','Class']
data2=pd.DataFrame(columns=[*names],index=wt)
for i in range(len(seizure_index)):
    data2.loc[wt[i],'RRIvar':'Class']=Raw_variance_RRI31_arr[seizure_index[i]],Raw_auto_RRI31_arr[seizure_index[i]],Raw_class_arr[i]
print(data2);print(data2.shape)
data=[data1,data2]
result = pd.concat(data,ignore_index=True)
print(result)
X=result.drop('Class',axis=1)
y=result['Class']
print(X);print(y)
y=y.astype('int')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train);print(X_test);
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
print(X_train);print(X_test);
### random forest classifier
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
pred_rfc=rfc.predict(X_test)
# print(pred_rfc)
## letls see how our model performed
print(classification_report(y_test,pred_rfc))
print(confusion_matrix(y_test,pred_rfc))
### SVM classifier
clf=svm.SVC()
clf.fit(X_train,y_train)
pred_clf=clf.predict(X_test)
print(classification_report(y_test,pred_clf))
print(confusion_matrix(y_test,pred_clf))
#### neural network
mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=200)
mlpc.fit(X_train,y_train)
pred_mlpc=mlpc.predict(X_test)
print(classification_report(y_test,pred_mlpc))
print(confusion_matrix(y_test,pred_mlpc))