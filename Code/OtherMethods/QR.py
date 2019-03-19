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
        l = l[:, 0]
        l = l.flatten()
        l = l.tolist()
        l = l[0]
        w2.append(l)
    return v1, w2

def householder_reflection(A):
    """Perform QR decomposition of matrix A using Householder reflection."""
    (num_rows, num_cols) = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)
    R = np.copy(A)

    # Iterative over column sub-vector and
    # compute Householder matrix to zero-out lower triangular matrix entries.
    for cnt in range(num_rows - 1):
        x = R[cnt:, cnt]

        e = np.zeros_like(x)
        t1 = np.linalg.norm(x)
        t2 = -A[cnt, cnt]
        t3 = copysign(np.linalg.norm(x), -A[cnt, cnt])

        e[0] = copysign(np.linalg.norm(x), -A[cnt, cnt])
        u = x + e
        v = u / np.linalg.norm(u)

        Q_cnt = np.identity(num_rows)
        t4 = np.outer(v, v)
        Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, v)

        R = np.dot(Q_cnt, R)
        t5 = Q_cnt.T
        Q = np.dot(Q, Q_cnt.T)

    return (Q, R)


def QR_transformation(A):
    start = time.clock()
    eigvals = []
    eigvecs = []
    a = A
    n = len(a)
    counter = 0
    counters = [10, 20, 30, 40, 50, 100]  # ,3000,4000,5000,6000,7000,8000,9000,10000]
    aaa = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(n):
            aaa[i][j] = a[i][j]

    P = np.identity(len(a))
    Pinv = np.identity(len(a))
    while (counter < 101):
        counter += 1
        q, r = np.linalg.qr(a)
        q1, r1 = householder_reflection(a)
        x = 0
        a = np.dot(np.array(np.linalg.inv(np.matrix(q))), a)
        a = np.dot(a, q)
        P = np.dot(np.array(np.linalg.inv(np.matrix(q))), P)
        Pinv = np.dot(Pinv, q)

        aa = np.dot(r, q)
        xx = a[0][n - 1]
        if counter in counters:
            ws = Pinv
            vs = []
            for i in range(n):
                vs.append(a[i][i])
            vs, ws = sort_eigens(vs, np.matrix(ws))
            eigvals.append(vs)
            eigvecs.append(ws)
        time1 = time.clock() - start

    return eigvals, eigvecs



def eigen_QR(G):
    matrix = nx.adjacency_matrix(G)
    V, W = QR_transformation(matrix.toarray())
    return V[0], W[0]




if __name__ == '__main__':
    datadir = "../../dataset/"
    graphNames = [ "karate","G100", "G200", "G500", "G1000", "G2000", "G5000", "G10000", "G25000", "G100000"]
    for graphname in graphNames:
        print (graphname)
        G = nx.read_edgelist(datadir+ graphname + ".txt")
        v, w =eigen_QR(G)
        x=0

