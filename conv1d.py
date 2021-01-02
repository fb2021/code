# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:07:29 2020

@author: faba
"""

import numpy as np
from keras import Sequential
from keras.layers import Dense,Dropout,Conv1D,MaxPooling1D,Flatten
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score)
#from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
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
                                                test_size = 0.15)
y_true = ytest
ytrain = to_categorical(ytrain, num_classes=2)
ytest = to_categorical(ytest, num_classes=2)
yval = to_categorical(yval, num_classes=2)


model = Sequential()

model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(64,1)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
#model.add(Dense(1024, activation='relu'))
model.add(Dense(2, activation='softmax'))

#Output Layer


#Compiling the neural network
model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

history=model.fit(xtrain, ytrain, batch_size=32, epochs=30,validation_data=(xval, yval))


#eval_model = model.evaluate(xtest, ytest)
#print("Test accuracy: ", eval_model[1])
score, acc = model.evaluate(xtest, ytest, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
y_pred = model.predict(xtest)
y_pred = model.predict_classes(xtest)
conf_mat = confusion_matrix(y_true, y_pred)
accuracy = 0
if float(np.sum(conf_mat))!=0:
    accuracy = float(conf_mat[0,0]+conf_mat[1,1])/float(np.sum(conf_mat))
recall = 0
if float(conf_mat[0,0]+conf_mat[1,0])!=0:
    recall = float(conf_mat[0,0])/float(conf_mat[0,0]+conf_mat[1,0])
precision = 0
if float(conf_mat[0,0]+conf_mat[0,1])!=0:
    precision = float(conf_mat[0,0])/float(conf_mat[0,0]+conf_mat[0,1])
#f1 score = 0
f1_score = 2*(float(precision*recall)/float(precision+recall))
print("confusion matrix")
print("----------------------------------------------")
print("accuracy")
print("%.6f" %accuracy)
print("racall")
print("%.6f" %recall)
print("precision")
print("%.6f" %precision)
print("f1score")
print("%.6f" %f1)

y_pred_proba = model.predict_proba(xtest)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_true,  y_pred_proba)
auc = metrics.roc_auc_score(y_true, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

##############################################################################3

# Get training and test loss histories
training_loss = history.history['accuracy']
test_loss = history.history['val_accuracy']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training acc', 'Test acc'])
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.show();


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


print("runtime:" + str(datetime.now()-start))
#print("runtime:" + str(time.time()-start))
#plt.plot(np.vstack([y_true, y_pred_proba]).T)
#plt.xlabel('Iteration #')
#plt.ylabel('Loss')








	