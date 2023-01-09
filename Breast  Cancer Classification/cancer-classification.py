# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.utils import plot_model

#Grabbing data
path = '..\Resources\cancer_classification.csv'
df = pd.read_csv(path)

#Showing basic information
print('---------')
print(df.head())
print('---------')
print(df.info())
print('---------')
print(df.describe())

#Creating a counterplot of benign and malignant
sns.set_style('darkgrid')
plt.figure(1)
sns.countplot(x='benign_0__mal_1',data=df)
plt.title('Count Plot of Benign and Malignant Classes')

#Viewing feature correlation with benign and malignant feature
print('---------')
bm_corr = df.corr()['benign_0__mal_1'][:-1].sort_values()
plt.figure(2)
bm_corr.plot(kind='bar')
plt.title('Correlation of Benign/Malignant Feature with Other Features')

plt.figure(3)
sns.heatmap(df.corr())

#--------------Splitting Data--------------
x = df.drop('benign_0__mal_1',axis=1).values
y = df['benign_0__mal_1'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=101)

#Scaling data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#--------------Model One--------------
#Creating model
model = Sequential()

model.add(Dense(30,activation='relu'))
model.add(Dense(15,activation='relu'))

#Output is sigmoid because this is binary classification
model.add(Dense(1,activation='sigmoid'))

#Compiling model
model.compile(optimizer='adam',loss='binary_crossentropy')

#Fitting model (intentionally have large number of epochs to show overfitting)
model.fit(x=x_train,y=y_train,epochs=600, validation_data=(x_test,y_test))

#Getting loss and validation loss as DataFrame
loss_df = pd.DataFrame(model.history.history)

#Plotting loss and validation loss
plt.figure(4)
sns.lineplot(data=loss_df)

#--------------Model Two--------------
#Creating model
model2 = Sequential()

model2.add(Dense(30,activation='relu'))
model2.add(Dense(15,activation='relu'))

#Output is sigmoid because this is binary classification
model2.add(Dense(1,activation='sigmoid'))

#Compiling model
model2.compile(optimizer='adam',loss='binary_crossentropy')

#Creating EarlyStopping object to stop fitting when minimum val_loss is reached
#, we are also waiting 25 epochs in case
early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience = 25)


#Fitting model (intentionally have large number of epochs to show overfitting)
model2.fit(x=x_train,y=y_train,epochs=600, validation_data=(x_test,y_test),
           callbacks=[early_stop])

#Getting loss and validation loss as DataFrame
loss_df = pd.DataFrame(model2.history.history)

#Plotting loss and validation loss
plt.figure(5)
sns.lineplot(data=loss_df)

#--------------Model Three--------------
#Creating model
model3 = Sequential()

model3.add(Dense(30,activation='relu'))
#Half of the 30 neurons, selected randomly, will turn off during each batch
model3.add(Dropout(0.5))

model3.add(Dense(15,activation='relu'))
#Half of the 15 neurons, selected randomly, will turn off
model3.add(Dropout(0.5))

#Output is sigmoid because this is binary classification
model3.add(Dense(1,activation='sigmoid'))

#Compiling model
model3.compile(optimizer='adam',loss='binary_crossentropy')


#Fitting model (intentionally have large number of epochs to show overfitting)
model3.fit(x=x_train,y=y_train,epochs=600, validation_data=(x_test,y_test),
           callbacks=[early_stop])

#Getting loss and validation loss as DataFrame
loss_df = pd.DataFrame(model3.history.history)

#Plotting loss and validation loss
plt.figure(6)
sns.lineplot(data=loss_df)

plot_model(model,to_file='cancer model1.png',
    show_shapes=True,
    show_dtype=True)

plot_model(model2,to_file='cancer model2.png',
    show_shapes=True,
    show_dtype=True)

plot_model(model3,to_file='cancer model3.png',
    show_shapes=True,
    show_dtype=True)

#--------------Prediction--------------
predictions = (model.predict(x_test) > 0.5)
print('---------')
print(classification_report(y_test,predictions))
print('---------')
print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

plt.figure(7)
sns.heatmap(cm,annot=True,)
plt.xlabel('Prediction')
plt.ylabel('True')


predictions = (model2.predict(x_test) > 0.5)
print('---------')
print(classification_report(y_test,predictions))
print('---------')
print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

plt.figure(8)
sns.heatmap(cm,annot=True,)
plt.xlabel('Prediction')
plt.ylabel('True')


predictions = (model3.predict(x_test) > 0.5)
print('---------')
print(classification_report(y_test,predictions))
print('---------')
print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

plt.figure(9)
sns.heatmap(cm,annot=True,)
plt.xlabel('Prediction')
plt.ylabel('True')


plt.show()