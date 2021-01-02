#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:48:54 2020

@author: mha
"""
from numpy import genfromtxt
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle

labels_csv = genfromtxt('labels.csv', delimiter=',',skip_header=0, names=True ,dtype=None, encoding=None)
nodes = np.load("nodes.npy")
vectors = np.load("vectors.npy")
labels = np.zeros(nodes.shape[0]) - 1

nodes_dict = { nodes[i] : i for i in range(len(nodes))}

for i in range(len(labels_csv)):
    if labels_csv['nodes'][i] in nodes:
        if labels_csv['label'][i] == 0:
            labels[nodes_dict[labels_csv['nodes'][i]]] = 0
        else:
            labels[nodes_dict[labels_csv['nodes'][i]]] = 1
#b_resize=vectors.reshape(vectors.shape[0],vectors.shape[1],1)
#vectors=b_resize            
#vectors,labels = shuffle(vectors,labels)
            
not_found = list(np.where(labels == -1)[0])
not_found.reverse()
#####################################################################
#z_scores_np = (vectors - vectors.mean()) / vectors.std()
 
# Min-Max scaling
 
#np_minmax = (vectors - vectors.min()) / (vectors.max() - vectors.min())
#b_resize=vectors.reshape(vectors.shape[0],vectors.shape[1],1)
#vectors=b_resize
#random.shuffle(vectors)
#print(vectors)
#temp = list(zip(vectors,labels))
#np.random.shuffle(temp)
#vectors, labels = zip(*temp)
#################################################
#vectors,labels = shuffle(vectors,labels)

#####################################################
for item in not_found:
    vectors = np.delete(vectors, item, axis = 0)
    labels = np.delete(labels, item, axis = 0)
    
xtrain, xtest, ytrain, ytest = train_test_split(vectors, labels, 
                                              test_size = 0.3)

np.save("xtrain",vectors)

np.save("ytrain",labels)




