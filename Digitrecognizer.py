# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:53:37 2022

@author: ASUS
"""
#importing libraries

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import pathlib
from keras.preprocessing import image

#%%
#importing data
data = pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#%%

#preprocessing data for training

X=data.copy()
X_test=test.copy()
print(X_test.head())

y=X.pop('label')
print(y.head())
print(X.shape)

id=X_test.pop('id')
print(X_test.head())

X_train,X_valid,y_train,y_valid=train_test_split(X,y,train_size=0.7)


X_train=np.array(X_train)
X_valid=np.array(X_valid)
X_test=np.array(X_test)


x=X_train[3].reshape(28,28)
test=X_test[10].reshape(28,28)
plt.imshow(test.astype('uint8'))


#%%

#creating a model
model=keras.Sequential([
    layers.Rescaling(1./255,input_shape=[784]),
    layers.Dense(128,activation='relu'),
    layers.Dense(10,activation='softmax')
    ])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer='adam',
              metrics=['accuracy'])
model.build()
model.summary()

#%%

#training the model
with tf.device('CPU/:0'):
    history=model.fit(X_train,y_train,validation_data=(X_valid,y_valid),epochs=15,
                  batch_size=60)


#%%

#visualization of the model's perfomance

import matplotlib.pyplot as plt
loss=history.history['loss']
val_loss=history.history['val_loss']

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
epoch=range(15)

plt.subplot(1,2,1)
plt.plot(epoch,acc,label='accuracy'),
plt.plot(epoch,val_acc,label='vaal accuracy')
plt.title('accuracy'),
#plt.legend(loc='upper left')

plt.subplot(1,2,2)
plt.plot(epoch,loss,label='loss')
plt.plot(epoch,val_loss,label='val loss')
plt.title('loss')
#plt.legend(loc='upper left')
#%%

#Saving the model

model.save('DigitReconizer1',save_format='h5')


#%%

#predicting the test data
prediction=model.predict(X_test)

#%%

#creating a csv for the predictions in test.csv
prediction=list(prediction)
predictions=[]

for i in range(0,len(prediction[0])):
    index = np.where(prediction[i] == max(prediction[i]))
    index=index[0].astype('int64')
    index=index[0]
    predictions.append(index)
    
output=pd.DataFrame({'id':id,'label':predictions})
print(output.head())

#csv file called submission.csv
output.to_csv('submission.csv',index=False)

    
#%%

#Predicting one image at time from test_images

#determining the path where the image is found

location=pathlib.Path('test images')
digitlist=list(location.glob('*'))

#loading the image
digit=image.load_img(digitlist[0],target_size=(28,28,1),color_mode="grayscale")
plt.imshow(digit)
#%%

#prepocessing the image
img_arr=(1/image.img_to_array(digit))*100
img_arr=np.array(img_arr)


#%%

#Enhancing the number's pixels for a clear image and recognizable data
for i in range(27):
    for j in range(27):
        img_arr[i][j][0] =round(img_arr[i][j][0])
        if img_arr[i][j][0]>1:
            img_arr[i][j][0] =img_arr[i][j][0]*80
#%%

#displaying the image
plt.imshow(img_arr.astype('uint8'))
            
#%%

#Expand the single image to batch size for the model to predict
img_batch=np.expand_dims(img_arr,axis=0)
img_batch=flatten(img_batch)           

#%%
#predicting the single image
pred=model.predict(img_batch)
print(pred)
index = np.where(pred[0] == max(pred[0]))
index=index[0].astype('int64')
index=index[0]

#%%

#printing the number corresponding to the picture
print(index)







