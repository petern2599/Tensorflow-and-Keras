# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from sklearn import metrics

#Grabbing data
path = '..\\Resources\\kc_house_data.csv'
df = pd.read_csv(path)

#Checking for any missing values
print(df.isnull().sum())

#Seeing head, info, and description of dataframe
print('---------')
print(df.head())
print('---------')
print(df.info())
print('---------')
print(df.describe())

#------------Exploring Data------------

#Creating distribution plot of the price
plt.figure(1)
sns.distplot(df['price'])
plt.title('Distribution Plot of House Prices')

#Creating countplot
plt.figure(2)
sns.countplot(df['bedrooms'])
plt.title('Count Plot of House Bedrooms')

#Creating heatmap of feature correlation
plt.figure(3)
sns.heatmap(df.corr())

#Observing feature that correlate with price
print(df.corr()['price'].sort_values())

plt.figure(4)
sns.scatterplot(x='price',y='sqft_living', data=df)
plt.title('Relationship of House Prices and Interior Living Space (Square Feet)')

plt.figure(5)
sns.boxplot(x='bedrooms',y='price',data=df)
plt.title('Relationship of Number of Bedrooms and House Prices')
plt.tight_layout(pad=1)

plt.figure(6)
fig, (ax1, ax2) = plt.subplots(ncols=2)
sns.scatterplot(x='price',y='long', data=df, ax=ax1)
sns.scatterplot(x='price',y='lat', data=df, ax=ax2)
plt.suptitle('Relationship of House Prices and Latitude vs Longitude')
plt.tight_layout(pad=1)


#Plotting scatterplot of locations based on price
plt.figure(8)
sns.scatterplot(x='long',y='lat',data=df,hue='price',palette='plasma',s=10)
plt.title('House Prices at Locations in King County')

#Dropping the top 1 percent of house data in terms of price
non_top_1_perc = df.sort_values('price',ascending=False).iloc[int(len(df)*0.01):]

plt.figure(9)
sns.scatterplot(x='long',y='lat',data=non_top_1_perc,hue='price',
                palette='plasma',s=10,edgecolor=None,alpha=0.2)
plt.title('House Prices at Locations in King County w/o Outliers')

#Creting boxplot of prices based on if house is at waterfront
plt.figure(10)
sns.boxplot(x='waterfront',y='price',data=df)
plt.title('House Prices Based on Houses Being at Waterfront')
plt.tight_layout(pad=1)

#Convert the date feature from string into datetime values
df['date'] = pd.to_datetime(df['date'])
print('---------')
print(df['date'])

#Extracting year and month into individual column (feature engineering)
df['year'] = df['date'].apply(lambda date:date.year)
df['month'] = df['date'].apply(lambda date:date.month)

#Dropping date column
df = df.drop('date',axis=1)

print('---------')
print(df.head())

#Observing any difference in house prices based on month

plt.figure(11)
df.groupby('month').mean()['price'].plot()
plt.ylabel('Average House Price')
plt.title('Average House Prices For Each Month')

#Observing any difference in house prices based on year
plt.figure(12)
df.groupby('year').mean()['price'].plot()
plt.title('Average House Prices For Each Year')

#Checking how many unique values in zipcode
print('---------')
print(df['zipcode'].value_counts())

#Too many unique zipcodes to use get_dummies function
df = df.drop('zipcode',axis=1)

#------------Splitting Data------------
#Using values because tensorflow doesn't work well with dataframes
x = df.drop('price',axis=1).values
y = df['price'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

#------------Creating Model------------

model = Sequential()

print('---------')
print(x_train.shape)

#Since there are 19 features, it is safe to use 19 neurons
#This might overkill and overfit the data a bit
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
#Setting up one neuron output layer since we want 1 feature
model.add(Dense(1))

#Compiling model
model.compile(optimizer='adam',loss='mse')

#Fitting model (uncomment for new model)
#model.fit(x=x_train,y=y_train,
#          validation_data=(x_test,y_test),
#          batch_size=128,epochs=400)

#Observing model for overfitting by comparing training loss vs validation loss
#loss_df = pd.DataFrame(model.history.history)
#loss_df.plot()

#Saving and loading model to save time from repeatedly fitting model
#model.save('regression_model.h5')

#Loading model
reg_model = load_model('regression_model.h5')

plot_model(reg_model,to_file='king_county_model.png',
    show_shapes=True,
    show_dtype=True)

#Predictions
predictions = reg_model.predict(x_test)
print('------------')
print("MAE: ", mean_absolute_error(y_test,predictions))
print("MSE: ", mean_squared_error(y_test,predictions))
print("RMSE: ", np.sqrt(mean_squared_error(y_test,predictions)))
print('------------')
print(df['price'].describe())

print('------------')
print('Variance Score: ',explained_variance_score(y_test, predictions))

#Concatenating predicted values and true values
pred = pd.Series(predictions.reshape(6480,))
true = pd.DataFrame(y_test)
output_df = pd.concat([true,pred],axis=1)
output_df.columns=['True','Predict']

#Plotting linear plot to compare true and predicted values
plt.figure(13)
sns.lmplot(x='True',y='Predict',data=output_df,line_kws={'color': 'red'})
plt.title('Linear Plot of True Values vs Predicted Values')
plt.tight_layout(pad=1)

#Printing r2_score
print('----------------')
print('R^2: ', metrics.r2_score(y_test,predictions))

plt.show()