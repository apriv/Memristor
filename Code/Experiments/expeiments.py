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







def find_Distance(v):
    min=10000000
    for i in range(1,len(v)):
        first = v[i-1]
        second = v[i]
        dist= first-second
        if first!=second:
            if abs(dist)<min:
                min=abs(dist)
   # min = math.ceil(min * 100) / 100
    min= round(min,10)
    if min<0.001:
        min=0.001
    return min

def randgen (size):
    obj = EigenPair()
    ep = 0.0001
    datadir = "/Users/sara 1/Documents/PHD/Python/"
    f = open(datadir + "dataset/" + "random "+size + ".txt", 'w')
    A = np.random.uniform(-1, 1, size * size).reshape(size,size)
    matrix = np.matrix(A)
    obj.update_parameters(matrix,isMatrix=true)
    obj.eigen_pairs(ep)
    obj.evaluate(ep)
    obj.log2(datadir, str(size), ep)

def readwrite():
    datadir = "/Users/sara 1/Documents/PHD/Python/dataset/air.txt"
    datadir2 = "/Users/sara 1/Documents/PHD/Python/dataset/air2.txt"

    f1 = open(datadir, 'r')
    f2 = open(datadir2, 'w')
    for line in f1:
        x = line.split(" ")
        f2.write(x[0] + '\t' + x[1] + '\n')

    f1.close()
    f2.close()


# static epsilon
def experiment1(datadir, graphNames):
    eps = [0.001]
    for graphname in graphNames:
        print ("****************graphname:  " + graphname)
        for ep in eps:
            print ("****************ep  :" + str(ep))
            obj = EigenPair()
            G = nx.read_edgelist(datadir + "dataset/" + graphname + ".txt")
            obj.update_parameters(G, isDynamic=False)
            obj.eigen_pairs(ep)
            obj.log2(datadir, graphname, ep)

    print ("finish")

# dynamic epsilon experiment
def experiment2(datadir, graphNames, k=-1):
    eps = [0.001]
    for graphname in graphNames:
        print ("****************graphname:  " + graphname)
        for ep in eps:
            print ("****************ep  :" + str(ep))
            obj = EigenPair()
            G = nx.read_edgelist(datadir + "dataset/" + graphname + ".txt")
            obj.update_parameters(G)
            obj.eigen_pairs(ep,k)
            if k ==-1:
                obj.log2(datadir, graphname, ep)
            else:
                obj.log3(datadir, graphname, ep)



    print ("finish")

# divideConqure standard
def experiment3(datadir, graphNames):
    ep = 0.0001
    ks = [10, 20]
    for graphname in graphNames:
        print ("****************graphname:  " + graphname)
        for k in ks:
            print ("****************ep  :" + str(ep))
            obj = EigenPair()
            obj.update_parameters(datadir, graphname)
            obj.eigen_pairs_divideConqure(ep, k, False)
            obj.evaluate(ep)
            obj.log4(datadir, graphname, k)

    print ("finish")

# divideConqure sweeping
def experiment4(datadir, graphNames,ks):
    ep = 0.0001

    for graphname in graphNames:
        print ("****************graphname:  " + graphname)
        for k in ks:
            print ("****************ep  :" + str(ep))
            obj = EigenPair()
            obj.update_parameters(datadir, graphname)
            obj.eigen_pairs_divideConqure(ep, k, True)
            obj.log5(datadir, graphname, k)

    print ("finish")

def experiment5(datadir, graphNames):
    for graphname in graphNames:
        print ("****************graphname:  " + graphname)
        obj = EigenPair()
        obj.update_parameters(datadir, graphname)
        obj.eigen_pairs_divideConqure(0.0001, 10, True)
        print("done1")
        obj.eigen_pairs_QR()
        print("done2")
        obj.log6(datadir, graphname, 0.0001)

def experiment6():
    x = EigenPair()
    a = np.zeros(shape=(4, 4))
    a[0][0] = 4
    a[0][1] = 1
    a[0][2] = -2
    a[0][3] = 2
    a[1][0] = 1
    a[1][1] = 2
    a[1][2] = 0
    a[1][3] = 1
    a[2][0] = -2
    a[2][1] = 0
    a[2][2] = 3
    a[2][3] = -2
    a[3][0] = 2
    a[3][1] = 1
    a[3][2] = -2
    a[3][3] = -1
    v2, w2 =  x.QR_transformation(a)
    v1, w1 = np.linalg.eig(np.matrix(a))
    v1, w1 = x.sort_eigens(v1,w1)

    for i in range(len(v1)):
        M1 = np.array([w1[i], w2[i]])
        similarity1 = cosine_similarity(M1)
        x = 0


    v1, w1 = x.sort_eigens(v1, w1)
    b = al.rmatrixqrunpackr(a,4,4)
    b2, P = x.householder_transformation3(a)

    a2 = np.zeros(shape=(4, 4))
    for i in range(len(a)):
        for j in range(len(a)):
            a2[i][j] = (a[i][j]) / 4
    v2, w2 = np.linalg.eig(np.matrix(a2))
    v2, w2 = x.sort_eigens(v2, w2)
    v3 = [4 * item for item in v2]

    datadir = "/Users/sara 1/Documents/PHD/Python/"
    G = nx.read_edgelist(datadir + "dataset/" + "karate" + ".txt")
    T = nx.adjacency_matrix(G)
    a = T.toarray()
    a2, P = x.householder_transformation3(a)
    v1, w1 = np.linalg.eig(np.matrix(a))
    v1, w1 = x.sort_eigens(v1, w1)
    v2, w2 = np.linalg.eig(np.matrix(a2))
    v2, w2 = x.sort_eigens(v2, w2)
    ws = np.dot(P, np.transpose(w2))
    ws = np.transpose(ws)
    ws = ws.tolist()

    for i in range(len(v1)):
        M1 = np.array([w1[i], ws[i]])
        similarity1 = cosine_similarity(M1)
        x = 0

    eps = [0.0001]
    for graphname in graphNames:
        print ("****************graphname:  " + graphname)
        for ep in eps:
            print ("****************ep  :" + str(ep))
            obj = EigenPair()
            obj.update_parameters(datadir, graphname)
            obj.eigen_pairs_divideConqure(ep, 4)
            obj.evaluate(ep)
            obj.log5(datadir, graphname, ep)

    print ("finish")

def experiment7(datadir, graphNames):
    eps = [1,0.75,0.5,0.25,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
    for graphname in graphNames:
        print ("****************graphname:  " + graphname)
        for ep in eps:
            print ("****************ep  :" + str(ep))
            obj = EigenPair()
            obj.update_parameters(datadir, graphname)
            obj.eigen_pairs_Distinct(ep)
            #obj.evaluate_Distinct(ep)
            obj.log_distinct(datadir, graphname, ep)

    print ("finish")


def experiment8(datadir, graphNames):
    eps = [0.001]
    for graphname in graphNames:
        print ("****************graphname:  " + graphname)
        for ep in eps:
            print ("****************ep  :" + str(ep))
            obj = EigenPair()
            G = nx.read_edgelist(datadir + "dataset/" + graphname + ".txt")
            matrix = nx.adjacency_matrix(G)
            A = matrix.toarray()
            #####
            B = A
            d = len(B)
            for i in range(d):
                rr = random.randint(0, d - 1)
                B[i][rr] = 1
                rr = random.randint(0, d - 1)
                B[i][rr] = 1
            A = B
            xx = np.random.rand(2, 2)
            xx[0][0] = 1
            xx[0][1] = -1
            xx[1][0] = 2
            xx[1][1] = 1
            #A = xx
            obj.update_parameters(A, isSymmetric=False, isMatrix=True)
            obj.eigen_pairs(ep)
            obj.evaluateASymmetric(ep)
            obj.log2(datadir, graphname, ep)

    print ("finish")


def fivepercentresults(datadir, graphNames):
    ep = 0.001
    percent = 100
    for graphname in graphNames:
        print ("****************graphname:  " + graphname)
       # print ("****************ep  :" + str(ep))
        obj = EigenPair()
        obj.update_parameters(datadir, graphname)
        obj.eigen_pairs_dynamic_percents(percent)
        obj.log_distinct_Percents(datadir, graphname, ep, percent)

    print ("finish")