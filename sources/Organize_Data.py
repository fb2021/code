#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:02:19 2020

@author: mha
"""
from numpy import genfromtxt
import numpy as np

graph = genfromtxt('G_Base_Line.csv', delimiter=',',skip_header=0, names=True ,dtype=None, encoding=None)
nodes = []
nodes.extend(graph['Node'])
nodes.extend(graph['Destination_Node'])
nodes = list(dict.fromkeys(nodes))
dictionary = { nodes[i]: i for i in range(len(nodes))}
for i in range(len(graph)):
    graph['Node'][i] = dictionary[graph['Node'][i]]
    graph['Destination_Node'][i] = dictionary[graph['Destination_Node'][i]]
np.save("nodes",nodes)
np.save("graph",graph)

