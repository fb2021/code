#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:21:34 2020

@author: mha
"""

import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
'''
import time
start=time.time()
'''
from datetime import datetime
start=datetime.now()


xtrain = np.load("xtrain.npy")
ytrain = np.load("ytrain.npy")

#ytrain = to_categorical(ytrain, num_classes=2)

xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, 
                                              test_size = 0.15)
xval, xtest, yval, ytest = train_test_split(xtrain, ytrain, 
                                                test_size = 0.1)
y_true = ytest
ytrain = to_categorical(ytrain, num_classes=2)
ytest = to_categorical(ytest, num_classes=2)
yval = to_categorical(yval, num_classes=2)


classifier = Sequential()

classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal', input_dim=64))
classifier.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
classifier.add(Dropout(0.3))
classifier.add(Dense(64, activation='relu', kernel_initializer='random_normal'))
classifier.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
classifier.add(Dropout(0.5))
classifier.add(Dense(64, activation='relu', kernel_initializer='random_normal'))
classifier.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
classifier.add(Dropout(0.3))
classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal', input_dim=64))
classifier.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
classifier.add(Dropout(0.3))
classifier.add(Dense(128, activation='relu'))
#Output Layer
classifier.add(Dense(2, activation='softmax'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

history=classifier.fit(xtrain, ytrain, batch_size=32, epochs=10,validation_data=(xval, yval))

#eval_model = classifier.evaluate(xtest, ytest)

#print("Test accuracy: ", eval_model[1])
score, acc = classifier.evaluate(xtest, ytest, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
y_pred = classifier.predict(xtest)
y_pred = classifier.predict_classes(xtest)
conf_mat = confusion_matrix(y_true, y_pred)
accuracy = 0
if float(np.sum(conf_mat))!=0:
    accuracy = float(conf_mat[0,0]+conf_mat[1,1])/float(np.sum(conf_mat))
#recall = 0
if float(conf_mat[0,0]+conf_mat[1,0])!=0:
    recall = float(conf_mat[0,0])/float(conf_mat[0,0]+conf_mat[1,0])
#precision = 0
if float(conf_mat[0,0]+conf_mat[0,1])!=0:
    precision = float(conf_mat[0,0])/float(conf_mat[0,0]+conf_mat[0,1])
#f1 score = 0
f1_score = 2*(float(precision*recall)/float(precision+recall))
print("confusion matrix")
print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("racall")
print("%.3f" %recall)
print("precision")
print("%.3f" %precision)
print("f1score")
print("%.3f" %f1)

y_pred_proba = classifier.predict_proba(xtest)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_true,  y_pred_proba)
auc = metrics.roc_auc_score(y_true, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
print("runtime:" + str(datetime.now()-start))
#print("runtime:" + str(time.time()-start))

