import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from numpy.linalg import *
from numpy import *
from math import sqrt
import networkx as nx
import math
import csv
import time
import random
from scipy import sparse
import numpy as np
from scipy.sparse import csc_matrix
from numpy import linalg as LA



##sort calculared eigens
def sort_eigens(v1, w1):
    dic = dict(zip(v1, range(len(v1))))
    v1 = sorted(v1, reverse=True)
    w2 = []
    for i in range(len(v1)):
        l = w1[:, dic[v1[i]]]
        #l = l[:, 0]
        l = l.flatten()
        l = l.tolist()
        #l = l[0]
        w2.append(l)
    return v1, w2

def power_iteration(A, num_simulations):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k, b_k1_norm

def shifted_power_iteration(A,num_simulations,num_eigs):
    vs=[]
    ws=[]

    for _ in range(num_eigs):
        v, w = power_iteration(A, num_simulations)
        vs.append(w)
        ws.append(v)
        v = np.array([v])
        B = np.dot(np.transpose(v),v)

        v_norm = np.linalg.norm(v)
        c= w / (v_norm*v_norm)
        C = c*B
        A = A- C
    return vs, ws


def eigen_power(G):
    A = nx.adjacency_matrix(G)
    A = A.toarray()
    V, W = shifted_power_iteration(A, 10, len(A))
    return V, W


##sort calculared eigens
def sort_eigens(v1, w1):
    dic = dict(zip(v1, range(len(v1))))
    v1 = sorted(v1, reverse=True)
    w2 = []
    for i in range(len(v1)):
        l = w1[:, dic[v1[i]]]
        #l = l[:, 0]
        l = l.flatten()
        l = l.tolist()
        #l = l[0]
        w2.append(l)
    return v1, w2
    V1, W1 = sort_eigens(V1, V)
    return V1, W1


if __name__ == '__main__':
    datadir = "../../dataset/"
    graphNames = [ "karate","G100", "G200", "G500", "G1000", "G2000", "G5000", "G10000", "G25000", "G100000"]
    for graphname in graphNames:
        print (graphname)
        G = nx.read_edgelist(datadir+ graphname + ".txt")
        V, W = eigen_power(G)

        x=0