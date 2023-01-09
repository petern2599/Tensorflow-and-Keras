# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score
from tensorflow.keras.utils import plot_model

data_dir = '..\\Resources\\cell_images'

#-----------Reading Data-----------

#Listing subdirectories
print('----------')
print(os.listdir(data_dir))

test_path = data_dir + '\\test'
train_path = data_dir + '\\train'

#Listing subdirectories
print('----------')
print(os.listdir(test_path))
print('----------')
print(os.listdir(train_path))

#Getting image data file name
print('----------')
print(os.listdir(train_path+'\\parasitized')[0])
print('----------')
print(os.listdir(train_path+'\\uninfected')[0])

#Parasitized cell file path
para_cell_path = train_path+'\\parasitized\\C100P61ThinF_IMG_20150918_144104_cell_162.png'
#Uninfected cell file path
uninfected_cell_path = train_path+'\\uninfected\\C100P61ThinF_IMG_20150918_144104_cell_128.png'

#Grabbing data from path using imread (BEWARE: returned imread data is automatically 
#normalized but the original image data is still 0-255)
para_cell_data = imread(para_cell_path)
uninfected_cell_data = imread(uninfected_cell_path)

#Plotting image
plt.figure(1)
plt.imshow(para_cell_data)
plt.title('Parasitized Cell')

plt.figure(2)
plt.imshow(uninfected_cell_data)
plt.title('Uninfected Cell')

#Checking out the length of data set
print('----------')
print('Length of Parasitized Training: ',len(os.listdir(train_path + '\\parasitized')))
print('----------')
print('Length of Uninfected Training: ',len(os.listdir(train_path + '\\uninfected')))
print('----------')
print('Length of Parasitized Test: ',len(os.listdir(test_path + '\\parasitized')))
print('----------') 
print('Length of Uninfected Test: ',len(os.listdir(test_path + '\\uninfected')))

#Checking if dimensions are the same
dim1 = []
dim2 = []

for img_filename in os.listdir(test_path+'\\uninfected'):
    img = imread(test_path + '\\uninfected\\' + img_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)


dim1_df = pd.DataFrame(dim1)
dim2_df = pd.DataFrame(dim2)
dim_df = pd.concat([dim1_df,dim2_df],axis=1)
dim_df.columns = ['Image Width','Image Height']
plt.figure(3)
sns.jointplot(x='Image Width',y='Image Height',data= dim_df)


#Dimensions are different, so using the average between both dimensions
print('----------')
print('Average of dim1: ', np.mean(dim1))
print('Average of dim2: ', np.mean(dim2))

#Defining image shape
img_shape = (130,130,3)

#-----------Processing Data-----------

#Creating ImageDataGenerator object to transform images to make the network
#more robust
img_gen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.1,
                             rescale=1/255,
                             zoom_range=0.1,
                             horizontal_flip=True,
                             fill_mode='nearest')

plt.figure(5)
plt.imshow(img_gen.random_transform(para_cell_data))
plt.title('Image Data Generator Transformed Image')

#-----------Creating Model-----------
model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=img_shape,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=img_shape,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=img_shape,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

print('----------')
print(model.summary())

early_stop = EarlyStopping(monitor='val_loss',patience=2)

print('----------')
#Creating a generator that changes image properties for training data
#Allow shuffling in training generator to reduce bias during model fitting
train_img_gen = img_gen.flow_from_directory(train_path,
                                            target_size=img_shape[:2],
                                            color_mode='rgb',
                                            batch_size=16,
                                            class_mode='binary',
                                            shuffle=True)
#Creating a generator that changes image properties for test data
test_img_gen = img_gen.flow_from_directory(test_path,
                                            target_size=img_shape[:2],
                                            color_mode='rgb',
                                            batch_size=16,
                                            class_mode='binary',
                                            shuffle=False)

print('----------')
print(train_img_gen.class_indices)

#Fitting model with generators (uncomment section to fit and get metrics)
results = model.fit(train_img_gen,epochs=20,validation_data=test_img_gen,
                    callbacks=[early_stop])

metrics = pd.DataFrame(model.history.history)

plt.figure(6)
sns.lineplot(data=metrics[['loss','val_loss']])

plt.figure(7)
sns.lineplot(data=metrics[['accuracy','val_accuracy']])

#model.save('malaria_detector_model.h5')

loaded_model = load_model('malaria_detector_model.h5')

#-----------Evaluating Model-----------
prediction_prob = loaded_model.predict(test_img_gen)
predictions = prediction_prob > 0.5

print('---------')
print(classification_report(test_img_gen.classes,predictions))
print('---------')
print(confusion_matrix(test_img_gen.classes,predictions))

cm = confusion_matrix(test_img_gen.classes,predictions)
plt.figure(8)
sns.heatmap(cm,annot=True,)
plt.xlabel('Prediction')
plt.ylabel('True')

#ROC curve
fpr, tpr, thresh = roc_curve(test_img_gen.classes, prediction_prob)
plt.figure(11)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

#AUC Score
print('---------')
print('AUC Score: ', roc_auc_score(test_img_gen.classes, prediction_prob))

plot_model(model,to_file='malaria model.png',
    show_shapes=True,
    show_dtype=True)

plt.show()