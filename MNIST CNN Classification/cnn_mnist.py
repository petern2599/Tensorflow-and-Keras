# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.utils import plot_model

#Loading mnist data
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#--------------Exploring Data--------------

#Grabbing a single image
single_img = x_train[0]

#Plotting image
plt.figure(1)
plt.imshow(single_img)

print('----------')
print('Single Image Label:', y_train[0])

#One-hot encoding integer labels into matrix
y_train_encode = to_categorical(y_train,num_classes=10)
y_test_encode = to_categorical(y_test,num_classes=10)

combined_labels = np.append(y_train,y_test)
combined_labels_df = pd.DataFrame(combined_labels)
combined_labels_df.columns = ['Labels']

plt.figure(2)
sns.countplot(x='Labels',data=combined_labels_df)
plt.title('Count Plot of Each Image Labels')

#We can do the same thing with get_dummies
y_train_dummy = pd.get_dummies(y_train)

#Normalize image data by dividing by 255 since future images will not exceed 
#an intensity value of 255
x_train = x_train/255
x_test = x_test/255

scaled_image = x_train[0]

#Plotting scaled image
plt.figure(3)
plt.imshow(scaled_image)

#Reshaping image data to batch_size,width,height,color_channels
#This is to let CNN know we are working with colors
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#--------------Creating Model--------------
model = Sequential()

#Add convolutional layer
#Arg1 -> number of filters (usually powers of 2)
#Arg2 -> kernel_size (usually 2x2,3x3,4x4)
#Arg3 -> stride distance
#Arg4 -> padding (use 'valid' if pixel_width/kernel_size [e.g 28/4] has no remainder)
#Arg5 -> input shape
model.add(Conv2D(filters=32,kernel_size=(4,4),strides=(1,1),padding='valid',
                 input_shape=(28,28,1),activation='relu'))

#Add Max Pool layer
#Arg1 -> pool size (usually 2x2 or half of the kernel_size)
model.add(MaxPool2D(pool_size=(2,2)))

#Add Flattening Layer
model.add(Flatten())

#Add Dense Layer
model.add(Dense(128,activation='relu'))

#Output Layer (Multi-Class -> Softmax)
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss',patience=1)

model.fit(x_train,y_train_encode,epochs=10,validation_data=(x_test,y_test_encode),
          callbacks=[early_stop])

#--------------Evaluating Model--------------

metrics = pd.DataFrame(model.history.history)

plt.figure(4)
sns.lineplot(data=metrics[['loss','val_loss']])

plt.figure(5)
sns.lineplot(data=metrics[['accuracy','val_accuracy']])

print('----------')
print('Model Training Evaluation:', model.evaluate(x_train,y_train_encode,verbose=0))
print('----------')
print('Model Validation Evaluation:', model.evaluate(x_test,y_test_encode,verbose=0))

#Grabbing probabilities of predicted class
predictions_prob = model.predict(x_test)
#Using np.argmax to get highest probability and grab column index
predictions = np.argmax(predictions_prob, axis=1)

print('---------')
print(classification_report(y_test,predictions))
print('---------')
print(confusion_matrix(y_test,predictions))

#Getting confusion matrix
cm = confusion_matrix(y_test,predictions)
plt.figure(6)
sns.heatmap(cm,annot=True,)
plt.xlabel('Prediction')
plt.ylabel('True')


plot_model(model,to_file='mnist model.png',
    show_shapes=True,
    show_dtype=True)

plt.show()