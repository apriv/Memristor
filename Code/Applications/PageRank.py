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




def pageRank(G, datadir=""):
    obj = SE.EigenPair()
    obj.update_parameters(G)
    start = time.clock()
    # vals, vecs, counters = obj.distinct_eigen_pairs_dynamic_percents(obj.A, 1.2, 0.9, 1,10)
    vals, vecs = obj.distinct_eigen_pairs(obj.A, 1, 0.9, 0.001, isbasic=False)

    result1 = obj.find_closest_eigen2(vals, 1)
    pagerank1 = vecs[result1]
    result2 = obj.find_closest_eigen2(obj.realVals, 1)
    pagerank2 = obj.realVecs[result2]
    v1 = pagerank1
    v2 = pagerank2

    error = 0
    c = 0
    for i in range(len(v1)):
        # if abs(v2[i]) > 0.001 :
        error += abs(abs(v1[i]) - abs(v2[i])) / abs(v2[i])
        c += 1
    error = error / c

    v1 = [item.real for item in v1]
    v2 = [item.real for item in v2]

    M = np.array([v1, v2])
    similarities = cosine_similarity(M)
    similarities2 = pairwise_distances(M)
    similarities3 = manhattan_distances(M)
    similarities4 = euclidean_distances(M)

    if datadir != "":
        with open(datadir + "output/" + graphname + "page.csv", 'wb') as csvfile:
            fieldnames = ["name", "result"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'name': cosine_similarity, 'result': str(similarities[0][1])})
            writer.writerow({'name': pairwise_distances, 'result': str(similarities2[0][1])})
            writer.writerow({'name': manhattan_distances, 'result': str(similarities3[0][1])})
            writer.writerow({'name': euclidean_distances, 'result': str(similarities4[0][1])})
            writer.writerow({'name': error, 'result': str(error)})







if __name__ == '__main__':
    datadir = "../../dataset/"
    graphNames = ["karate", "G100", "G200", "G500", "G1000", "G2000", "G5000", "G10000", "G25000", "G100000"]
    for graphname in graphNames:
        G = nx.read_edgelist(datadir  + graphname + ".txt")
        pageRank(G, 10)
        x=0

