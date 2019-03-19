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
import sys
sys.path.insert(0, '../SweepingMethod/')
import EigenPair as SE



def feature_selection(datadir, alpha=0.5):
    n = 500
    filepath1 = datadir + 'data.txt'
    filepath2 = datadir + 'label.txt'
    Xs = []
    Xs2 = []
    for i in range(n):
        Xs.append([])
    Ys = []
    labels = []

    with open(filepath1) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            Y = []
            splited = line.split(" ")
            for i in range(n):
                item = int(splited[i])
                Xs[i].append(item)
                Y.append(item)
            line = fp.readline()
            cnt += 1
            Ys.append(Y)

    with open(filepath2) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            labels.append(int(line))
            line = fp.readline()
            cnt += 1
            Ys.append(Y)
    Xs1 = []
    mus2 = []
    Xs2 = []
    mus = []
    deltas = []
    fs = []
    m = []
    for items in Xs:
        neg = []
        pos = []
        x = []
        max1 = max(items)
        for i in range(len(items)):
            item = float(items[i]) / max1
            x.append(item)
            if labels[i] == 1:
                pos.append(item)
            else:
                neg.append(item)
        Xs1.append(x)
        mus2.append(np.std(x))
        Xs2.append((pos, neg))
        mu1 = mean(pos)
        mu2 = mean(neg)
        delta1 = np.std(pos)
        delta2 = np.std(neg)
        mus.append((mu1, mu2))
        deltas.append((delta1, delta2))
        fs.append(float(abs(mu1 - mu2)) / ((delta1 * delta1) + (delta2 * delta2)))
        s1 = sum(pos)
        s2 = sum(neg)
        s = float(s1) + float(s2)
        a1 = s1 * math.log((s1) / (s + float(len(pos)) / (len(pos) + len(neg))))
        a2 = s2 * math.log((s2) / (s + float(len(neg)) / (len(pos) + len(neg))))
        m.append(a1 + a2)

    fs = np.reshape(fs, (n, 1))
    m = np.reshape(m, (1, n))
    A = np.matmul(fs, m)

    for i in range(n):
        for j in range(n):
            A[i][j] = alpha * A[i][j] + (1 - alpha) * (max(mus2[i], mus2[j]))

    obj = SE.EigenPair()
    obj.update_parameters(A, isMatrix=True)
    vals1, vecs1 = obj.distinct_eigen_pairs(A, obj.maximum, obj.minimum, 0.001,  isbasic=False , k=1)

    x = 0


if __name__ == '__main__':
    datadir = "../../dataset/"
    result = feature_selection(datadir)
    x=0