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
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import mean_squared_error

#Grabbing data
path = '..\\Resources\\RSCCASN.csv'
df = pd.read_csv(path,parse_dates=True,index_col='DATE')
df.columns = ['Sales']

print('--------')
print(df.head())
print('--------')
print(df.info())
print('--------')
print(df.describe())

plt.figure(1)
sns.lineplot(data=df)

test_size_months = 18
test_index = len(df) - test_size_months

train = df.iloc[:test_index]
test = df.iloc[test_index:]

#Normalize data
scaler = MinMaxScaler()

scaled_train = scaler.fit_transform(train)
scaled_test = scaler.transform(test)

#-----------Creating Model-----------

length=12

generator = TimeseriesGenerator(scaled_train,scaled_train,
                                length=length,batch_size=1)

n_features = 1

model = Sequential()

model.add(LSTM(100,activation='relu',input_shape=(length,n_features)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

print('--------')
print(model.summary())

early_stop = EarlyStopping(monitor = 'val_loss', patience = 2)

validation_generator = TimeseriesGenerator(scaled_test,scaled_test,
                                           length=length,batch_size=1)

model.fit(generator,epochs=20,
          validation_data=validation_generator,
          callbacks=[early_stop])

#-----------Evaluating Model-----------
#Checking model losses
losses = pd.DataFrame(model.history.history)

plt.figure(2)
losses.plot()

#Initializing predictions
test_predictions = []

#Grabbing first batch of training data
first_eval_batch = scaled_train[-length:]
#Reshape the dimension of the batch
current_batch = first_eval_batch.reshape((1,length,n_features))

for i in range(len(test)):
    #Predict next data point
    current_prediction = model.predict(current_batch)[0]
    
    #Append prediction
    test_predictions.append(current_prediction)
    
    #Remove first data point in batch and append current prediction to batch
    current_batch = np.append(current_batch[:,1:,:],[[current_prediction]],axis=1)
    
#Reverse the scaling of prediction
true_predictions = scaler.inverse_transform(test_predictions)

#Convert predictions to dataframe and combine it with test data
true_predictions = pd.DataFrame(true_predictions,index=test.index)
compare_df = pd.concat([test,true_predictions],axis=1)
compare_df.columns = ['Sales',"LSTM Prediction"]

plt.figure(3)
compare_df.plot()

plot_model(model,to_file='sales model.png',
    show_shapes=True,
    show_dtype=True)

#-----------Creating Model for Whole Data-----------

full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)

generator = TimeseriesGenerator(scaled_full_data,scaled_full_data,
                                length=length,batch_size=1)

model = Sequential()

model.add(LSTM(100,activation='relu',input_shape=(length,n_features)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

model.fit(generator,epochs=8)

#-----------Forecasting Data-----------
forecast = []
periods = 12

first_eval_batch = scaled_full_data[-length:]
current_batch = first_eval_batch.reshape((1,length,n_features))

for i in range(periods):
    current_prediction = model.predict(current_batch)[0]
    
    forecast.append(current_prediction)
    
    current_batch = np.append(current_batch[:,1:,:],[[current_prediction]],axis=1)
    
true_forecast = full_scaler.inverse_transform(forecast)
forecast_index = pd.date_range(start='2019-11-01',periods=periods,
                               freq='MS')
true_forecast = pd.DataFrame(true_forecast,index=forecast_index)
true_forecast.columns = ['Forecast']

#-----------Evaluating Forecast-----------
plt.figure(5)
ax=df.plot()
true_forecast.plot(ax=ax)

plt.figure(6)
ax=df.plot()
true_forecast.plot(ax=ax)
plt.xlim('2018-01-01','2020-12-01')

plot_model(model,to_file='sales forecast model.png',
    show_shapes=True,
    show_dtype=True)

print('----------')
print('RSME: ', np.sqrt(mean_squared_error(compare_df['Sales'],compare_df['LSTM Prediction'])))
plt.show()
