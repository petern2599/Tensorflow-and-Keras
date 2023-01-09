# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.utils import plot_model

#Loading fashion mnist data
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

plt.figure(1)
plt.imshow(x_train[0])
print('----------')
print('x_train[0] label: ',y_train[0])

#Normalizing image data
x_train = x_train/255
x_test = x_test/255

#Reshaping dimension of image data
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#Performing one-hot encoding for labels
y_train_enc = to_categorical(y_train,num_classes=10)
y_test_enc = to_categorical(y_test,num_classes=10)


#Checking labels
combined_labels = np.append(y_train,y_test)
combined_labels_df = pd.DataFrame(combined_labels)
combined_labels_df.columns = ['Labels']

plt.figure(2)
sns.countplot(x='Labels',data=combined_labels_df)
plt.title('Count Plot of Each Image Label')

#-----------Creating Model-----------
model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(4,4),strides=(1,1),padding='valid',
                 input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(10,activation='softmax'))

#Compiling model
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',
              metrics=['accuracy'])

print('----------')
print(model.summary())

early_stop = EarlyStopping(monitor='val_loss',patience=2)

#Fitting model
model.fit(x_train,y_train_enc,epochs=10,validation_data=(x_test,y_test_enc),
          callbacks=[early_stop])

#-----------Evaluating Model-----------
#Evalutating losses
metrics = pd.DataFrame(model.history.history)

plt.figure(3)
sns.lineplot(data=metrics[['loss','val_loss']])

plt.figure(4)
sns.lineplot(data=metrics[['accuracy','val_accuracy']])

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
plt.figure(5)
sns.heatmap(cm,annot=True,)
plt.xlabel('Prediction')
plt.ylabel('True')

plot_model(model,to_file='fashion mnist model.png',
    show_shapes=True,
    show_dtype=True)

plt.show()