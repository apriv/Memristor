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


def conditionNumber(G,datadir=""):
    obj = SE.EigenPair()
    matrix = nx.adjacency_matrix(G)
    A = matrix.toarray()
    B = np.negative(A)
    obj.update_parameters(A, isMatrix=True)
    vals1, vecs1 = obj.distinct_eigen_pairs(A, obj.maximum, obj.minimum, 0.001, isbasic=False, k=1)
    vals2, vecs2 = obj.distinct_eigen_pairs(B, -(obj.minimum), -(obj.maximum), 0.001, isbasic=False, k=1)

    CN = abs(vals1[0]) / abs(vals2[0])
    cnReal = abs(obj.realVals[0]) / abs(obj.realVals[obj.dimension - 1])

    error = abs(CN - cnReal) / abs(cnReal)

    if datadir != "":
        with open(datadir + "conditionNumber.csv", 'wb') as csvfile:
            fieldnames = ["name", "result"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'name': "cn", 'result': str(CN)})
            writer.writerow({'name': "cnreal", 'result': str(cnReal)})
            writer.writerow({'name': "error", 'result': str(error)})


    return CN, error



if __name__ == '__main__':
    datadir = "../../dataset/"
    graphNames = [ "karate","G100", "G200", "G500", "G1000", "G2000", "G5000", "G10000", "G25000", "G100000"]
    for graphname in graphNames:
        print (graphname)
        G = nx.read_edgelist(datadir+ graphname + ".txt")
        V, W = conditionNumber(G)
        x=0