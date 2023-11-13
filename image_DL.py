import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import layers,datasets,models

(training_images,training_labels),(testing_images,testing_labels)=datasets.cifar10.load_data()
training_images,testing_images=training_images/255,testing_images/255

class_names=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i],cmap=plt.cm.binary)
    plt.xlabel([class_names[training_labels[i][0]]])
plt.show()

# training_images=training_images[:20000]
# training_labels=training_labels[:20000]
# testing_images=testing_images[:4000]
# testing_labels=testing_labels[:4000]
#
#
# model=models.Sequetial()
# model.add(layers.Cov2D(32,(3,3),activatio='relu',input_shape=(32,32,3)))
# model.add(layers.MaxPoolig2D((2,2)))
# model.add(layers.Cov2D(64,(3,3),activatio='relu'))
# model.add(layers.MaxPoolig2D((2,2)))
# model.add(layers.Cov2D(64,(3,3),activatio='relu'))
# model.add(layers.Flatte())
# model.add(layers.Dese(64),activatio='relu')
# model.add(layers.Dese(10),activatio='softmax')
#
#
# model.compile(optimizer='adam',loss='sparse_categorical crosssentropy',metrics=['accuracy'])
#
# model.fit(training_images,training_labels,epoch=10,validatio_data=(testing_images,testing_labels))
#
# loss,accuracy=models.evaluate(testing_images,testing_labels)
#
# model.save('image_classifier.model')
#
#
#
# model=models.load_model('image_classifier.model')
#
# img=cv2.imread('horse.jpg')
# img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
#
# prediction=model.predict(np.array([img])/255)
# index=np.argmax(prediction)
# print(f'Prediction is {class_names[index]}')







