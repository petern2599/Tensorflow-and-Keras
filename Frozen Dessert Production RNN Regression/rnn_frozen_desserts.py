# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import mean_squared_error
from tensorflow.keras.utils import plot_model

path = '..\\Resources\\Frozen_Dessert_Production.csv'
df = pd.read_csv(path,parse_dates=True,index_col='DATE')
df.columns = ['Production']

print('--------')
print(df.head())
print('--------')
print(df.info())
print('--------')
print(df.describe())

plt.figure(1)
df.plot()

test_size_months = 24
test_index = len(df)-test_size_months

train = df.iloc[:test_index]
test = df.iloc[test_index:]

scaler = MinMaxScaler()

scaled_train = scaler.fit_transform(train)
scaled_test = scaler.transform(test)


#-----------Creating Model-----------
length = 23
n_features = 1

generator = TimeseriesGenerator(scaled_train,scaled_train,
                                length=length,batch_size=1)

model = Sequential()
model.add(LSTM(100,activation='relu',input_shape=(length,n_features)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

validation_generator = TimeseriesGenerator(scaled_test,scaled_test,
                                length=length,batch_size=1)

early_stop = EarlyStopping(monitor='val_loss',patience=2)

model.fit(generator,epochs=20,validation_data=validation_generator,
          callbacks=[early_stop])

#-----------Evaluating Model-----------
losses = pd.DataFrame(model.history.history)

plt.figure(3)
sns.lineplot(data=losses)

predictions = []

#Grabbing first batch of training data
first_eval_batch = scaled_test[-length:]
#Reshape the dimension of the batch
current_batch = first_eval_batch.reshape((1,length,n_features))

for i in range(len(test)):
    #Predict next data point
    current_prediction = model.predict(current_batch)[0]
    
    #Append prediction
    predictions.append(current_prediction)
    
    #Remove first data point in batch and append current prediction to batch
    current_batch = np.append(current_batch[:,1:,:],[[current_prediction]],axis=1)

#Reverse the scaling of prediction
true_prediction = scaler.inverse_transform(predictions)

#Convert predictions to dataframe and combine it with test data
true_prediction = pd.DataFrame(true_prediction,index=test.index)
compare_df = pd.concat([test,true_prediction],axis=1)
compare_df.columns=['Production','LSTM Predictions']

plt.figure(4)
compare_df.plot()

print('--------')
print('RSME:', np.sqrt(mean_squared_error(compare_df['Production'],compare_df['LSTM Predictions'])))

plot_model(model,to_file='frozen desserts model.png',
    show_shapes=True,
    show_dtype=True)
plt.show()