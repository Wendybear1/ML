from __future__ import division
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd



# conduct CNN on raw EEG
# dataset_1 = pd.read_csv("C:/Users/wxiong/Documents/PHD/2021.10/ML_preictal/60min/ES/data_ES_preictal_60min_CNN.csv")
dataset_1 = pd.read_csv("C:/Users/wxiong/Documents/PHD/2021.10/ML_preictal/ES_preictal/ES_merged.csv")
print(len(dataset_1))
# print(dataset_1);print(len(dataset_1))
data_sum=[]
for i in range(int(len(dataset_1)/4)):
# for i in range(2):
    cell=dataset_1.iloc[[0+4*i,1+4*i,2+4*i,3+4*i],[1,2,3,4,5]]
    data_sum.append([cell.values.tolist(),1])
    # data_sum.append(cell.values.tolist())

# dataset_2 = pd.read_csv("C:/Users/wxiong/Documents/PHD/2021.10/ML_preictal/60min/PNES/data_PNES_preictal_60min_CNN.csv")
dataset_2 = pd.read_csv("C:/Users/wxiong/Documents/PHD/2021.10/ML_preictal/PNES_preictal/PNES_merged.csv")
print(len(dataset_2))
for i in range(int(len(dataset_2)/4)):
# for i in range(2):
    cell=dataset_2.iloc[[0+4*i,1+4*i,2+4*i,3+4*i],[1,2,3,4,5]]
    data_sum.append([cell.values.tolist(),0])
    # data_sum.append(cell.values.tolist())
print(data_sum)
# # print(type(data_sum))
# # print(data_sum[1])
x_sum=[]
y_sum=[]
for m in range(len(data_sum)):
    x_sum.append(data_sum[m][0])
    y_sum.append(data_sum[m][1])
# print(x_sum);print(y_sum);

val_acc_arr=[]
for k in range(10):
    X_train, X_validation, Y_train, Y_validation = train_test_split(x_sum, y_sum, test_size=0.20, random_state=1)
    # print(X_train);print(X_validation);
    # print(Y_train);print(Y_validation);

    X_train=np.asarray(X_train)
    X_validation=np.asarray(X_validation)
    Y_train=np.asarray(Y_train)
    Y_validation=np.asarray(Y_validation)

    # X_train=tf.keras.utils.normalize(X_train, axis=1)
    # X_validation = tf.keras.utils.normalize(X_validation, axis=1)
    # print(X_train);
    # print(len(X_train));print(Y_train)
    # print(X_train.shape);
    # print(X_validation.shape);

    # # model configuration
    batch_size=4
    loss_function='binary_crossentropy'
    no_classes=2
    no_epochs=10
    validation_split = 0.2
    verbosity = 1
    input_shape = (4, 5, 1)

    X_train=X_train.reshape(X_train.shape[0], 4, 5, 1)
    X_validation=X_validation.reshape(X_validation.shape[0], 4, 5, 1)
    # print(X_train)

    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(8, kernel_size=(2, 2), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2),padding='same'))
    # model.add(tf.keras.layers.Conv2D(4, kernel_size=(2, 2), activation='relu', padding='same'))
    # model.add(layers.MaxPooling2D((2, 2),padding='same'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(8, activation='relu'))

    model.add(tf.keras.layers.Dense(no_classes, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=no_epochs, validation_data=(X_validation, Y_validation))
    val_loss, val_acc = model.evaluate(X_validation, Y_validation, verbose=0)
    val_acc_arr.append(val_acc)

# print(val_acc_arr)
df = pd.DataFrame()
df['CNN']=val_acc_arr
df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\CS_CNN_60min_ALL_EEGperformance_Accuracy.csv')




# ### critical slowing CNN
# import pandas as pd
# # conduct CNN
# val_acc_arr=[]
# for m in range(100):
#     dataset_1 = pd.read_csv("C:/Users/wxiong/Documents/PHD/combine_features/preictal test/backup/60min/PNES_preictal_and_ES_preictal (60min).csv")
#     print(len(dataset_1))
#     dataset_ES=dataset_1[dataset_1['type'] == 1]
#     dataset_PNES=dataset_1[dataset_1['type'] == 0]
#     print(len(dataset_ES));print(len(dataset_PNES))
#
#     data_sum = []
#     # for i in range(9440):
#     for i in range(9381):
#         row = dataset_ES.iloc[i]
#         # print(row)
#         cell = [[
#             [[row['F7_EEGvar'], row['F7_EEGauto']], [row['F3_EEGvar'], row['F3_EEGauto']],
#              [row['Fz_EEGvar'], row['Fz_EEGauto']], [row['F4_EEGvar'], row['F4_EEGauto']],
#              [row['F8_EEGvar'], row['F8_EEGauto']]],
#             [[row['T3_EEGvar'], row['T3_EEGauto']], [row['C3_EEGvar'], row['C3_EEGauto']],
#              [row['Cz_EEGvar'], row['Cz_EEGauto']], [row['C4_EEGvar'], row['C4_EEGauto']],
#              [row['T4_EEGvar'], row['T4_EEGauto']]],
#             [[row['T5_EEGvar'], row['T5_EEGauto']], [row['P3_EEGvar'], row['P3_EEGauto']],
#              [row['Pz_EEGvar'], row['Pz_EEGauto']], [row['P4_EEGvar'], row['P4_EEGauto']],
#              [row['T6_EEGvar'], row['T6_EEGauto']]],
#             [[0, 0], [row['O1_EEGvar'],row['O1_EEGauto']], [0, 0], [row['O2_EEGvar'],row['O2_EEGauto']], [0, 0]]],
#             1]
#
#         data_sum.append(cell)
#     # print(data_sum);print(len(data_sum));
#
#
#
#     for i in range(6018):
#         row = dataset_PNES.iloc[i]
#         # print(row)
#         cell = [[
#             [[row['F7_EEGvar'], row['F7_EEGauto']], [row['F3_EEGvar'], row['F3_EEGauto']],
#              [row['Fz_EEGvar'], row['Fz_EEGauto']], [row['F4_EEGvar'], row['F4_EEGauto']],
#              [row['F8_EEGvar'], row['F8_EEGauto']]],
#             [[row['T3_EEGvar'], row['T3_EEGauto']], [row['C3_EEGvar'], row['C3_EEGauto']],
#              [row['Cz_EEGvar'], row['Cz_EEGauto']], [row['C4_EEGvar'], row['C4_EEGauto']],
#              [row['T4_EEGvar'], row['T4_EEGauto']]],
#             [[row['T5_EEGvar'], row['T5_EEGauto']], [row['P3_EEGvar'], row['P3_EEGauto']],
#              [row['Pz_EEGvar'], row['Pz_EEGauto']], [row['P4_EEGvar'], row['P4_EEGauto']],
#              [row['T6_EEGvar'], row['T6_EEGauto']]],
#             [[0, 0], [row['O1_EEGvar'],row['O1_EEGauto']], [0, 0], [row['O2_EEGvar'],row['O2_EEGauto']], [0, 0]]],
#             0]
#
#         data_sum.append(cell)
#     # print(data_sum);print(len(data_sum));
#
#
#     x_sum=[]
#     y_sum=[]
#     for m in range(len(data_sum)):
#         x_sum.append(data_sum[m][0])
#         y_sum.append(data_sum[m][1])
#     # print(x_sum);print(y_sum);
#     # print(len(x_sum));print(len(y_sum));
#
#
#     X_train, X_validation, Y_train, Y_validation = train_test_split(x_sum, y_sum, test_size=0.20, random_state=1)
#     # print(X_train);print(X_validation);
#     # print(Y_train);print(Y_validation);
#
#     X_train=np.asarray(X_train)
#     X_validation=np.asarray(X_validation)
#     Y_train=np.asarray(Y_train)
#     Y_validation=np.asarray(Y_validation)
#
#     # X_train=tf.keras.utils.normalize(X_train, axis=1)
#     # print(X_train);print(len(X_train)); print(type(X_train))
#     # print(X_train.shape)
#
#     # model configuration
#     batch_size=4
#     loss_function='binary_crossentropy'
#     no_classes=2
#     no_epochs=3
#     validation_split = 0.2
#     verbosity = 1
#     # input_shape = (4, 5, 1)
#
#
#     model=tf.keras.models.Sequential()
#
#     model.add(tf.keras.layers.Conv2D(4, kernel_size=(2, 2), activation='relu',padding='same'))
#     model.add(tf.keras.layers.Conv2D(4, kernel_size=(2, 2), activation='relu',padding='same'))
#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.Dense(8, activation='relu'))
#     model.add(tf.keras.layers.Dense(no_classes,activation=tf.nn.softmax))
#     model.compile(optimizer= 'adam',loss=loss_function, metrics=['accuracy'])
#
#
#     model.fit(X_train,Y_train,batch_size=batch_size,epochs=no_epochs)
#     val_loss, val_acc=model.evaluate(X_validation,Y_validation,verbose=0)
#     print(val_loss, val_acc)
#     val_acc_arr.append(val_acc)
#
# df = pd.DataFrame()
# df['CNN']=val_acc_arr
# df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\CNN_60min_ALL_EEGperformance_Accuracy.csv')