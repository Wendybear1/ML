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


## create a dataframe
# data={'ID':['SA0124','QLD0098','QLD0227','VIC1202'],
#       'age':['20','25','30','51'],
#       'gender':['F','M','M','F'],
#       'No.seizure':['17','8','5','13'],
#       'type':['F','F','F','G']}
# df=pd.DataFrame(data,columns=['ID','age','gender','No.seizure','type'])
# print(df)



## case1
data = pd.read_csv('C:/Users/wxiong/Documents/PHD/self learning/MLtext.csv')
df = pd.DataFrame(data, columns = ['total','training','testing','type','Circadian model AUC','Critical slowing model AUC','Combined model AUC','gender','age', 'duration',])
pd.set_option('max_columns', None)
print(data)
# df1= pd.DataFrame(data, columns = ['Circadian model AUC','Critical slowing model AUC','Combined model AUC'])
# print(df1)
print(data.info())
print(data.isnull().sum())



##preprocessing data
bins=(0,0.5,1)
groups_names=['bad','good']
data['Critical slowing model AUC']=pd.cut(data['Critical slowing model AUC'],bins=bins,labels=groups_names)
pd.set_option('max_columns', None)
print(data)
# data['Critical slowing model AUC'].unique()
# print(data['Critical slowing model AUC'].unique())
# pd.set_option('max_columns', None)
# print(data)
le=LabelEncoder()
data['Critical slowing model AUC']=le.fit_transform(data['Critical slowing model AUC'])
data['gender']=le.fit_transform(data['gender'])
data['type']=le.fit_transform(data['type'])

pd.set_option('max_columns', None)
print(data)
# data['Circadian model AUC']=pd.cut(data['Circadian model AUC'],bins=bins,labels=groups_names)
# data['Combined model AUC']=pd.cut(data['Combined model AUC'],bins=bins,labels=groups_names)
# performance_quality=LabelEncoder()
# data['Circadian model AUC']=performance_quality.fit_transform(data['Circadian model AUC'])
# performance_quality=LabelEncoder()
# data['Combined model AUC']=performance_quality.fit_transform(data['Combined model AUC'])
# pd.set_option('max_columns', None)
# print(data)

print(data['Critical slowing model AUC'].value_counts())
sns.countplot(data['Critical slowing model AUC'])
plt.show()

X=data.drop('Critical slowing model AUC',axis=1)
y=data['Critical slowing model AUC']
print(X);print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train);print(X_test);

#applying standard scaling to get optimised result
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
print(X_train);print(X_test);





### random forest classifier
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
pred_rfc=rfc.predict(X_test)
print(pred_rfc)
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




