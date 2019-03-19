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



def modularity1(A, group):
    n = len(A)
    B = np.zeros(shape=(n, n))
    nodes = range(n)
    degres = [0] * n
    for i in range(n):
        degres[i] = sum(A[i])
    m = sum(degres)

    for i in nodes:
        for j in nodes:
            B[i][j] = A[i][j] - ((degres[i]) * (degres[j])) / float(m)

    v, w = np.linalg.eig(B)
    leading = w[0]

    group1 = []
    group2 = []
    for i in group:
        if leading[i] < 0:
            group1.append(i)
        else:
            group2.append(i)

    return B, group1, group2

def modularity2(B, group, group1, group2, dic):
    result = []
    n = len(group)
    B1 = np.zeros(shape=(n, n))
    ii = -1
    for i in group:
        sigma = 0
        ii += 1
        for k in group:
            sigma += B[dic[i]][dic[k]]
        jj = -1
        for j in group:
            jj += 1
            if (i in group1 and j in group1) or (i in group2 and j in group2):
                delta = -1
            else:
                delta = 1
            B1[ii][jj] = B[dic[i]][dic[j]] - (delta * sigma)

    # v, w = np.linalg.eig(B1)
    obj = SE.EigenPair()
    obj.update_parameters(B1, isMatrix=True)
    # obj.eigen_pairs_dynamic(ep)
    vals1, vecs1 = obj.distinct_eigen_pairs(B1, obj.maximum, obj.minimum, 0.001, isbasic=False, k=1)

    leading = vecs1[0]

    group11 = []
    group12 = []
    for i in group1:
        if leading[dic[i]] < 0:
            group11.append(i)
        else:
            group12.append(i)

    if len(group11) > 0:
        if len(group12) > 0:
            B2, dic2 = modularity3(B1, group1, dic)
            result.append(modularity2(B2, group1, group11, group12, dic2))
        else:
            result.append(group11)
            # final_result.append(group11)
    else:
        result.append(group12)
        # final_result.append(group12)

    group21 = []
    group22 = []
    for i in group2:
        if leading[dic[i]] < 0:
            group21.append(i)
        else:
            group22.append(i)

    if len(group21) > 0:
        if len(group22) > 0:
            B2, dic2 = modularity3(B1, group2, dic)
            result.append(modularity2(B1, group2, group21, group22, dic2))
        else:
            result.append(group21)
            # final_result.append(group21)

    else:
        result.append(group22)
        # final_result.append(group22)

    return result

def modularity3(B, group, dic):
    n = len(group)
    B1 = np.zeros(shape=(n, n))
    dic2 = {}
    ii = -1
    for i in group:
        ii += 1
        jj = -1
        dic2.update({i: ii})
        for j in group:
            jj += 1
            B1[ii][jj] = B[dic[i]][dic[j]]
    return B1, dic2

def modularity_sweep(G):
    output = nx.Graph()
    nodes = []
    dic = {}
    counter = 0
    for node in G.nodes():
        nodes.append(counter)
        dic.update({node: counter})
        counter += 1
    output.add_nodes_from(nodes)
    for edge in G.edges():
        output.add_edge(dic[edge[0]], dic[edge[1]])
    A = nx.adjacency_matrix(output)
    A = A.toarray()
    group = output.nodes()
    B, group1, group2 = modularity1(A, group)
    dic = dict(zip(range(len(output.nodes())), output.nodes()))
    r = modularity2(B, group, group1, group2, dic)
    label=0
    counter = 0
    while(len(r)>0):
        items = r[counter]
        r.remove(items)
        Flag = False
        for item in items:
            if type(item) == type(0):
                dic[item] = label
                Flag = True
            else:
                r.append(item)
        if Flag:
            label +=1



    return dic





if __name__ == '__main__':
    datadir = "../../dataset/"
    graphNames = [ "karate","G100", "G200", "G500", "G1000", "G2000", "G5000", "G10000", "G25000", "G100000"]
    for graphname in graphNames:
        print (graphname)
        G = nx.read_edgelist(datadir+ graphname + ".txt")
        mod = modularity_sweep(G)
        x=0