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

def Lanczos( A, v, m=100 ):
    n = len(v)
    if m>n: m = n;
    # from here https://en.wikipedia.org/wiki/Lanczos_algorithm
    V = np.zeros( (m,n) )
    T = np.zeros( (m,m) )
    vo   = np.zeros(n)
    beta = 0
    for j in range( m-1 ):
        w    = np.dot( A, v )
        alfa = np.dot( w, v )
        w    = w - alfa * v - beta * vo
        beta = np.sqrt( np.dot( w, w ) )
        vo   = v
        v    = w / beta
        T[j,j  ] = alfa
        T[j,j+1] = beta
        T[j+1,j] = beta
        V[j,:]   = v
    w    = np.dot( A,  v )
    alfa = np.dot( w, v )
    w    = w - alfa * v - beta * vo
    T[m-1,m-1] = np.dot( w, v )
    V[m-1]     = w / np.sqrt( np.dot( w, w ) )
    return T, V


def eigen_lanczos(G):
    A = nx.adjacency_matrix(G)
    A = A.toarray()
    n = len(A)
    # ---- approximate solution by Lanczos
    v0 = np.random.rand(n)
    v0 /= np.sqrt(np.dot(v0, v0))
    T, V = Lanczos(A, v0, m=len(A))
    esT, vsT = np.linalg.eig(T)
    VV = np.dot(V, np.transpose(V))  # check orthogonality
    V1=[]
    for i in range (n):
        V1.append(T[i,i])
    V1, W1 = sort_eigens(V1, V)
    return V1, W1

if __name__ == '__main__':
    datadir = "../../dataset/"
    graphNames = [ "karate","G100", "G200", "G500", "G1000", "G2000", "G5000", "G10000", "G25000", "G100000"]
    for graphname in graphNames:
        print (graphname)
        G = nx.read_edgelist(datadir+ graphname + ".txt")
        V, W = eigen_lanczos(G)

        x=0

