# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:51:47 2022

@author: peter
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report,confusion_matrix,roc_curve, roc_auc_score
from tensorflow.keras.utils import plot_model

path = '..//Resources//lending_club_loan_two.csv'
df = pd.read_csv(path)

print('----------')
print(df.head())
print('----------')
print(df.info())
print('----------')
print(df.describe())

#--------------Exploring Data--------------

#Observing the amount of people who fully paid loan vs those that were charged off
plt.figure(1)
sns.countplot(df['loan_status'])
plt.title('Count Plot of Loan Status')

#Checking out correlation
print('----------')
print(df.corr())

plt.figure(2)
sns.heatmap(df.corr(),annot=True)

plt.figure(3)
sns.scatterplot(x='loan_amnt',y='installment',data=df)
plt.title('Relationship of Loan Amount and Installments')

plt.figure(4)
sns.boxplot(x='loan_status',y='loan_amnt',data=df)
plt.title('Loan Amount Based on Loan Status')
plt.tight_layout(pad=1)

plt.figure(5)
sns.countplot(x='grade',data=df,hue='loan_status',order = sorted(df['grade'].unique()))
plt.title('Count Plot of Loan Grades Based on Loan Status')
plt.tight_layout(pad=1)

#Ordering subgrade for plotting countplot
plt.figure(6)
sns.countplot(x='sub_grade',data=df,hue='loan_status',order = sorted(df['sub_grade'].unique()))
plt.title('Count Plot of Loan Sub-Grades Based on Loan Status')
plt.tight_layout(pad=1)

#--------------Missing Data--------------
#Checking out features with missing values
print('----------')
print(df.isna().sum())

#Checking out number of unique employment titles
print('----------')
print(df['emp_title'].value_counts())

#There's too many unique titles so dropping feature
df = df.drop('emp_title',axis=1)


emp_len_order = ['< 1 year','1 year']
for num in range(2,10):
    emp_len_order.append(f'{num} years')
emp_len_order.append('10+ years')

#Checking out employment length
plt.figure(7)
sns.countplot(x='emp_length',data=df, hue='loan_status', order=emp_len_order)
plt.title('Count Plot of Employment Length Based on Loan Status')
plt.tight_layout(pad=1)

#Getting percentage between loan status based on employment length
emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']
emp_len_perc = emp_co/emp_fp

plt.figure(8)
emp_len_perc.plot(kind='bar')
plt.title('Count Plot of Employment Length Percentage of Charged Off to Fully Paid Loans')
plt.tight_layout(pad=1)

#Since there is no significant different between loan statuses for each length
#dropping the employment length feature
df = df.drop('emp_length',axis=1)

#Checking out the difference between title and purpose features since details
#look similar
print('----------')
print(df['purpose'].head(10))
print('----------')
print(df['title'].head(10))

#Since the title and purpose features are pretty similar and since the title 
#feature is missing values, dropping title column
df = df.drop('title',axis=1)

#Checking if other features correlate to the number of mortgage accounts
print('----------')
print(df.corr()['mort_acc'].sort_values())

#Grabbing the mean of the number of mortgage accounts based on total number of credit lines
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
print('----------')
print(total_acc_avg)

#Filling missing values for the number of mortgage accounts with the average based
#on total number of credit lines
def fill_mort_acc(total_acc,mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.
    
    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc
    
df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

#Since the revol_util and pub_rec_bankruptcies features are missing a small amount
#compared to the length of the dataset, rows with these missing values are dropped
df = df.dropna()

#--------------Preprocessing Categorical Data--------------
#Replacing string value with binary classification for loan status
df['loan_status'] = df['loan_status'].apply(lambda stat:1 if stat=='Fully Paid' else 0)

#Grabbing all non-numeric features
print('----------')
print(df.select_dtypes(['object']).columns)

#Changing the values for term into numeric values
df['term'] = df['term'].apply(lambda x:int(x[:3]))

#Since the grade feature is explain in the sub_grade feature, dropping the grade feature
df = df.drop('grade',axis=1)

#Use get_dummies on sub_grade feature and concatenating to dataframe
sub_grade_dummy = pd.get_dummies(df['sub_grade'],drop_first=True)
df = df.drop('sub_grade',axis=1)
df = pd.concat([df,sub_grade_dummy],axis=1)

#Replacing NONE and ANY values in home ownership feature into OTHER, due to its
#redundancy and these values doesn't appear often in this feature
df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

#Use get_dummies on home_ownership feature and concatenating to dataframe
home_dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,home_dummies],axis=1)

#Use get_dummies on verification_status feature and concatenating to dataframe
verif_dummies = pd.get_dummies(df['verification_status'],drop_first=True)
df = df.drop('verification_status',axis=1)
df = pd.concat([df,verif_dummies],axis=1)

#Dropping issue_d feature because it doesn't make sense to know the date the loan
#is funded before considering them for the loan
df = df.drop('issue_d',axis=1)

#Use get_dummies on purpose feature and concatenating to dataframe
purp_dummies = pd.get_dummies(df['purpose'],drop_first=True)
df = df.drop('purpose',axis=1)
df = pd.concat([df,purp_dummies],axis=1)

#Extracting the year from the earliest_cr_line feature and adding it to dataframe
#since the month is not so useful by itself and pairing it with the year creates
#too many entries to use with get_dummies
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))

#Dropping the earliest_cr_line feature since its no longer needed
df = df.drop('earliest_cr_line',axis=1)

#Use get_dummies on initial_list_status feature and concatenating to dataframe
init_list_stat_dummies = pd.get_dummies(df['initial_list_status'],drop_first=True)
df = df.drop('initial_list_status',axis=1)
df = pd.concat([df,init_list_stat_dummies],axis=1)

#Use get_dummies on application_type feature and concatenating to dataframe
app_type_dummies = pd.get_dummies(df['application_type'],drop_first=True)
df = df.drop('application_type',axis=1)
df = pd.concat([df,app_type_dummies],axis=1)

#For the address feature, can't use the entire address with get_dummies, instead
#will use zip code 
df['zip_code'] = df['address'].apply(lambda address:address[-5:])
df = df.drop('address',axis=1)

#Use get_dummies on initial_list_status feature and concatenating to dataframe
zip_dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop('zip_code',axis=1)
df = pd.concat([df,zip_dummies],axis=1)

#--------------Splitting Data--------------
x = df.drop('loan_status',axis=1)
y = df['loan_status']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#--------------Creating Model--------------
model = Sequential()

model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(39,  activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(18,  activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy')

early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience = 25)

#--------------Fitting Data--------------

#Fitting model(uncomment if making new model)
#model.fit(x=x_train,y=y_train, epochs=25, batch_size=256, 
#          validation_data=(x_test, y_test),callbacks=[early_stop])

#loss_df = pd.DataFrame(model.history.history)

#plt.figure(9)
#sns.lineplot(data=loss_df)

#model.save('project_model.h5')  

#Loading model
loaded_model = load_model('project_model.h5')

plot_model(loaded_model,to_file='loan model.png',
    show_shapes=True,
    show_dtype=True)

#Making predictions
predictions = (loaded_model.predict(x_test) > 0.5)
print('---------')
print(classification_report(y_test,predictions))
print('---------')
print(confusion_matrix(y_test,predictions))

#Creating confusion matrix
cm = confusion_matrix(y_test,predictions)

plt.figure(10)
sns.heatmap(cm,annot=True,)
plt.xlabel('Prediction')
plt.ylabel('True')

#ROC curve
predictions_proba = (loaded_model.predict(x_test))
fpr, tpr, thresh = roc_curve(y_test, predictions_proba)
plt.figure(11)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

#AUC Score
print('---------')
print('AUC Score: ', roc_auc_score(y_test, predictions_proba))


plt.show()