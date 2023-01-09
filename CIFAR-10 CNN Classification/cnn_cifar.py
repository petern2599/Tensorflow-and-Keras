# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.utils import plot_model

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

plt.figure(1)
plt.imshow(x_train[1])

print('----------')
print('Single Image Label:', y_train[1])

#Normalizing data
x_train = x_train/255
x_test = x_test/255

#One-hot encoding integer labels into matrix
y_train_encode = to_categorical(y_train,num_classes=10)
y_test_encode = to_categorical(y_test,num_classes=10)

#Checking image labels
combined_labels = np.append(y_train,y_test)
combined_labels_df = pd.DataFrame(combined_labels)
combined_labels_df.columns = ['Labels']

plt.figure(3)
sns.countplot(x='Labels',data=combined_labels_df)
plt.title('Count Plot of Each Image Label')

#--------------Creating Model--------------

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4,4),strides=(1,1),padding='valid',
                 input_shape=(32,32,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(4,4),strides=(1,1),padding='valid',
                 input_shape=(32,32,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

print('----------')
print(model.summary())

early_stop = EarlyStopping(monitor='val_loss',patience=2)

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

plot_model(model,to_file='cifar10 model.png',
    show_shapes=True,
    show_dtype=True)
plt.show()