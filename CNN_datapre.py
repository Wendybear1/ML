from __future__ import division
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras import layers


# # # ### prepare for classifiers
# col=['Fz_EEGvar', 'C4_EEGvar', 'Pz_EEGvar', 'C3_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'P4_EEGvar',
#             'P3_EEGvar', 'T4_EEGvar', 'T3_EEGvar', 'O2_EEGvar', 'O1_EEGvar', 'F7_EEGvar', 'F8_EEGvar',
#             'T6_EEGvar', 'T5_EEGvar', 'Cz_EEGvar',
#             'Fz_EEGauto', 'C4_EEGauto', 'Pz_EEGauto', 'C3_EEGauto', 'F3_EEGauto', 'F4_EEGauto', 'P4_EEGauto',
#             'P3_EEGauto', 'T4_EEGauto', 'T3_EEGauto', 'O2_EEGauto', 'O1_EEGauto', 'F7_EEGauto', 'F8_EEGauto',
#             'T6_EEGauto', 'T5_EEGauto', 'Cz_EEGauto']
#
#
# #Epileptic seizure raw
# X_matrice = []
# Y_matrice = []
# directory = r'/Users/wxiong/Documents/PHD/combine_features/ES_with_tags_15min'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d: d.path)
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset_whole = pd.read_csv(entry.path, sep=',', skipinitialspace=True)
#         dataset = dataset_whole[dataset_whole['tags'] == 2]
#         for m in range(int(len(dataset['Fz_EEGvar']) / (4))):
#             # for m in range(2):
#             matrix = []
#             for n in range(17):
#                 a = [
#                      [dataset[col[n]].values[0 + 4 * m], dataset[col[n]].values[1 + 4 * m], dataset[col[n]].values[2 + 4 * m],dataset[col[n]].values[3 + 4 * m]],
#                      [dataset[col[n + 17]].values[0 + 4 * m], dataset[col[n + 17]].values[1 + 4 * m], dataset[col[n + 17]].values[2 + 4 * m],dataset[col[n + 17]].values[3 + 4 * m]],
#                      # [dataset['ch31_RRIvar'].values[0 + 4 * m], dataset['ch31_RRIvar'].values[1 + 4 * m],dataset['ch31_RRIvar'].values[2 + 4 * m], dataset['ch31_RRIvar'].values[3 + 4 * m]],
#                      # [dataset['ch31_RRIauto'].values[0 + 4 * m], dataset['ch31_RRIauto'].values[1 + 4 * m],dataset['ch31_RRIauto'].values[2 + 4 * m], dataset['ch31_RRIauto'].values[3 + 4 * m]]
#                      ]
#
#                 matrix.append(a)
#
#             X_matrice.append(matrix)
#             Y_matrice.append(1)
#
#
#
#
# directory = r'/Users/wxiong/Documents/PHD/combine_features/PNES_with_tags_15min'
# dir_list = list(os.scandir(directory))
# dir_list.sort(key=lambda d: d.path)
# for entry in dir_list:
#     if (entry.path.endswith(".csv")) and entry.is_file():
#         dataset_whole = pd.read_csv(entry.path, sep=',', skipinitialspace=True)
#         dataset = dataset_whole[dataset_whole['tags'] == 2]
#         for m in range(int(len(dataset['Fz_EEGvar']) / (4))):
#             # for m in range(2):
#             matrix = []
#             for n in range(17):
#                 a = [
#                      [dataset[col[n]].values[0 + 4 * m], dataset[col[n]].values[1 + 4 * m], dataset[col[n]].values[2 + 4 * m],dataset[col[n]].values[3 + 4 * m]],
#                      [dataset[col[n + 17]].values[0 + 4 * m], dataset[col[n + 17]].values[1 + 4 * m], dataset[col[n + 17]].values[2 + 4 * m],dataset[col[n + 17]].values[3 + 4 * m]],
#                      # [dataset['ch31_RRIvar'].values[0 + 4 * m], dataset['ch31_RRIvar'].values[1 + 4 * m],dataset['ch31_RRIvar'].values[2 + 4 * m], dataset['ch31_RRIvar'].values[3 + 4 * m]],
#                      # [dataset['ch31_RRIauto'].values[0 + 4 * m], dataset['ch31_RRIauto'].values[1 + 4 * m],dataset['ch31_RRIauto'].values[2 + 4 * m], dataset['ch31_RRIauto'].values[3 + 4 * m]]
#                      ]
#
#                 matrix.append(a)
#
#             X_matrice.append(matrix)
#             Y_matrice.append(0)
#
#
#
# X_arr = np.array(X_matrice)
# y_arr = np.array(Y_matrice)
#
#
# val_acc_arr=[]
# for k in range(100):
#     X_train, X_validation, Y_train, Y_validation = train_test_split(X_arr, y_arr, test_size=0.20, random_state=1)
#
#     # X_train=np.asarray(X_train)
#     # X_validation=np.asarray(X_validation)
#     # Y_train=np.asarray(Y_train)
#     # Y_validation=np.asarray(Y_validation)
#
#
#     X_train=tf.keras.utils.normalize(X_train, axis=1)
#     X_validation = tf.keras.utils.normalize(X_validation, axis=1)
#     # print(X_train);print( Y_train)
#     # print(len(X_train));print(Y_train)
#     # print(X_train.shape);
#     # print(X_validation.shape);
#
#     # # model configuration
#     batch_size=5
#     loss_function='binary_crossentropy'
#     no_classes=2
#     no_epochs=10
#     validation_split = 0.2
#     verbosity = 1
#     # input_shape = (17, 4, 4)
#     input_shape = (17, 2, 4)
#     print(X_train.shape[0])
#
#     # X_train=X_train.reshape(X_train.shape[0], 17, 4, 4, 1)
#     # X_validation=X_validation.reshape(X_validation.shape[0], 17, 4, 4, 1)
#     # print(X_train)
#
#     model=tf.keras.models.Sequential()
#     model.add(tf.keras.layers.Conv2D(8, kernel_size=(2, 2), activation='relu', padding='same', input_shape=input_shape))
#     model.add(layers.MaxPooling2D((2, 2),padding='same'))
#     # model.add(tf.keras.layers.Conv2D(4, kernel_size=(2, 2), activation='relu', padding='same'))
#     # model.add(layers.MaxPooling2D((2, 2),padding='same'))
#
#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.Dense(8, activation='relu'))
#
#     model.add(tf.keras.layers.Dense(no_classes, activation=tf.nn.softmax))
#     model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
#
#     model.fit(X_train, Y_train, epochs=no_epochs, validation_data=(X_validation, Y_validation))
#     val_loss, val_acc = model.evaluate(X_validation, Y_validation, verbose=0)
#     val_acc_arr.append(val_acc)
#
# print(val_acc_arr)
# df = pd.DataFrame()
# df['CNN']=val_acc_arr
# df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\CNN\\CS_CNN_15min_30min_ALL_EEGperformance_Accuracy.csv')







# # ### prepare for classifiers
col=['Fz_EEGvar', 'C4_EEGvar', 'Pz_EEGvar', 'C3_EEGvar', 'F3_EEGvar', 'F4_EEGvar', 'P4_EEGvar',
            'P3_EEGvar', 'T4_EEGvar', 'T3_EEGvar', 'O2_EEGvar', 'O1_EEGvar', 'F7_EEGvar', 'F8_EEGvar',
            'T6_EEGvar', 'T5_EEGvar', 'Cz_EEGvar',
            'Fz_EEGauto', 'C4_EEGauto', 'Pz_EEGauto', 'C3_EEGauto', 'F3_EEGauto', 'F4_EEGauto', 'P4_EEGauto',
            'P3_EEGauto', 'T4_EEGauto', 'T3_EEGauto', 'O2_EEGauto', 'O1_EEGauto', 'F7_EEGauto', 'F8_EEGauto',
            'T6_EEGauto', 'T5_EEGauto', 'Cz_EEGauto']


#Epileptic seizure raw
X_matrice = []
Y_matrice = []
directory = r'/Users/wxiong/Documents/PHD/combine_features/preictal test/15min/preictal'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d: d.path)
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        dataset_whole = pd.read_csv(entry.path, sep=',', skipinitialspace=True)
        dataset = dataset_whole[dataset_whole['type'] == 1]
        for m in range(int(len(dataset['Fz_EEGvar']) / (4))):
            # for m in range(2):
            matrix = []
            for n in range(17):
                a = [
                     [dataset[col[n]].values[0 + 4 * m], dataset[col[n]].values[1 + 4 * m], dataset[col[n]].values[2 + 4 * m],dataset[col[n]].values[3 + 4 * m]],
                     [dataset[col[n + 17]].values[0 + 4 * m], dataset[col[n + 17]].values[1 + 4 * m], dataset[col[n + 17]].values[2 + 4 * m],dataset[col[n + 17]].values[3 + 4 * m]],
                     # [dataset['ch31_RRIvar'].values[0 + 4 * m], dataset['ch31_RRIvar'].values[1 + 4 * m],dataset['ch31_RRIvar'].values[2 + 4 * m], dataset['ch31_RRIvar'].values[3 + 4 * m]],
                     # [dataset['ch31_RRIauto'].values[0 + 4 * m], dataset['ch31_RRIauto'].values[1 + 4 * m],dataset['ch31_RRIauto'].values[2 + 4 * m], dataset['ch31_RRIauto'].values[3 + 4 * m]]
                     ]

                matrix.append(a)

            X_matrice.append(matrix)
            Y_matrice.append(1)




directory = r'/Users/wxiong/Documents/PHD/combine_features/preictal test/15min/preictal'
dir_list = list(os.scandir(directory))
dir_list.sort(key=lambda d: d.path)
for entry in dir_list:
    if (entry.path.endswith(".csv")) and entry.is_file():
        dataset_whole = pd.read_csv(entry.path, sep=',', skipinitialspace=True)
        dataset = dataset_whole[dataset_whole['type'] == 0]
        for m in range(int(len(dataset['Fz_EEGvar']) / (4))):
            # for m in range(2):
            matrix = []
            for n in range(17):
                a = [
                     [dataset[col[n]].values[0 + 4 * m], dataset[col[n]].values[1 + 4 * m], dataset[col[n]].values[2 + 4 * m],dataset[col[n]].values[3 + 4 * m]],
                     [dataset[col[n + 17]].values[0 + 4 * m], dataset[col[n + 17]].values[1 + 4 * m], dataset[col[n + 17]].values[2 + 4 * m],dataset[col[n + 17]].values[3 + 4 * m]],
                     # [dataset['ch31_RRIvar'].values[0 + 4 * m], dataset['ch31_RRIvar'].values[1 + 4 * m],dataset['ch31_RRIvar'].values[2 + 4 * m], dataset['ch31_RRIvar'].values[3 + 4 * m]],
                     # [dataset['ch31_RRIauto'].values[0 + 4 * m], dataset['ch31_RRIauto'].values[1 + 4 * m],dataset['ch31_RRIauto'].values[2 + 4 * m], dataset['ch31_RRIauto'].values[3 + 4 * m]]
                     ]

                matrix.append(a)

            X_matrice.append(matrix)
            Y_matrice.append(0)



X_arr = np.array(X_matrice)
y_arr = np.array(Y_matrice)


val_acc_arr=[]
for k in range(100):
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_arr, y_arr, test_size=0.20, random_state=1)

    # X_train=np.asarray(X_train)
    # X_validation=np.asarray(X_validation)
    # Y_train=np.asarray(Y_train)
    # Y_validation=np.asarray(Y_validation)


    X_train=tf.keras.utils.normalize(X_train, axis=1)
    X_validation = tf.keras.utils.normalize(X_validation, axis=1)
    # print(X_train);print( Y_train)
    # print(len(X_train));print(Y_train)
    # print(X_train.shape);
    # print(X_validation.shape);

    # # model configuration
    batch_size=5
    loss_function='binary_crossentropy'
    no_classes=2
    no_epochs=10
    validation_split = 0.2
    verbosity = 1
    # input_shape = (17, 4, 4)
    input_shape = (17, 2, 4)
    print(X_train.shape[0])

    # X_train=X_train.reshape(X_train.shape[0], 17, 4, 4, 1)
    # X_validation=X_validation.reshape(X_validation.shape[0], 17, 4, 4, 1)
    # print(X_train)

    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(8, kernel_size=(2, 2), activation='relu', padding='same', input_shape=input_shape))
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

print(val_acc_arr)
df = pd.DataFrame()
df['CNN']=val_acc_arr
df.to_csv(f'C:\\Users\\wxiong/Documents\\PHD\\combine_features\\performances\\Preictal_classify\\CNN\\CS_CNN_15min_ALL_EEGperformance_Accuracy_test.csv')