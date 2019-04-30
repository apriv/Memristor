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

    for i in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)  # value

        # re normalize the vector
        b_k = b_k1 / b_k1_norm  # vector

    return b_k, b_k1_norm


def shifted_power_iteration(A, num_simulations,num_eigs):
    vs = []
    ws = []
    A = A

    for _ in range(num_eigs):
        print(A)
        v, w = power_iteration(A, num_simulations)
        print(v, w)
        vs.append(w)
        ws.append(v)
        v = np.array([v])
        B = np.dot(np.transpose(v), v)

        v_norm = np.linalg.norm(v)
        c= w / (v_norm*v_norm)
        C = c*B
        A = A - C
    return vs, ws


def eigen_power(G):
    A = nx.adjacency_matrix(G)
    A = A.toarray()
    V, W = shifted_power_iteration(A, 10, len(A))
    return V, W


# sort calculated eigens
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
    # find leading ev
    #print(power_iteration(np.array([[1, -3, 2], [-3, -8, 2], [2, 2, 3]]), 5))

    # find all ev
    #V, _ = shifted_power_iteration(np.array([[1, 2, 3], [2, 4, 5], [3, 4, 6]]) , 10, 3)
    #print(V)

    V, _ = shifted_power_iteration(np.array([[4, 5], [6, 5]]), 10, 2)
    print(V)

    '''
    datadir = "../../dataset/"
    graphNames = [ "karate","G100", "G200", "G500", "G1000", "G2000", "G5000", "G10000", "G25000", "G100000"]
    graphNames = [ "karate","G100", "G200", "G500", "G1000"]
    for graphname in graphNames:
        time_start = time.time()
        print (graphname)
        G = nx.read_edgelist(datadir+ graphname + ".txt")
        V, W = eigen_power(G)
        x=0
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
    '''

