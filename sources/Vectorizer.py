#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:02:19 2020

@author: mha
"""

import networkx as nx
from node2vec import Node2Vec
import numpy as np

# FILES
EMBEDDING_FILENAME = './embeddings.emb'
EMBEDDING_MODEL_FILENAME = './embeddings.model'
nodes = np.load("nodes.npy")
#nodes = { i : nodes[i] for i in range(len(nodes))}
g = np.load("graph.npy")
graph = nx.Graph()
for i in range(len(g)):
    graph.add_edge(int(g['Node'][i]), int(g['Destination_Node'][i]), weight=g['Weight'][i] )


# Precompute probabilities and generate walks
#node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)

## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
# Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="./tmp")

# Embed
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
#model.wv.most_similar('2')  # Output node names are always strings

# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
model.save(EMBEDDING_MODEL_FILENAME)

vectors = []
for i in range(len(nodes)):
    vectors.append(model.wv.get_vector(str(i)))

vectors = np.asarray(vectors)

np.save("vectors", vectors)