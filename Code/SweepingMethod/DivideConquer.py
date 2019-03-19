from numpy import *
import networkx as nx
import EigenPair
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



def eigen_pairs_divideConqure(G, ep=0.001, size=10, bool1=False):
    start = time.clock()
    obj = EigenPair.EigenPair()
    obj.update_parameters(G)
    T2 = obj.A
    ###
    v1= obj.realVals
    w1 = obj.realVecs
    #v1, w1 = obj.sort_eigens(v1, w1)
    a22, Pt, P = obj.householder_transformation4(T2)
    v3, w3 = np.linalg.eig(a22)
    v3, w3 = obj.sort_eigens2(v3, w3)
    ws = np.dot(Pt, np.transpose(w3))
    ws = ws.transpose()
    #for i in range(len(v1)):
     #   M1 = np.array([w1[i], ws[i]])
      #  similarity1 = cosine_similarity(M1)
       # x = 0
    ####

    T, PH, P = obj.householder_transformation4(T2)
    v1, w1 = obj.divideConqure(T, size, bool1, ep)
    w1 = np.dot(PH, np.transpose(w1))
    w1 = np.transpose(w1)
    w1 = w1.tolist()
    obj.vals = v1
    obj.vecs = w1
    obj.time = time.clock() - start

    return obj.vals, obj.vecs









if __name__ == '__main__':
    datadir = "../../dataset/"
    graphNames = [ "karate","G100", "G200", "G500", "G1000", "G2000", "G5000", "G10000", "G25000", "G100000"]
    for graphname in graphNames:
        print (graphname)
        G = nx.read_edgelist(datadir+ graphname + ".txt")
        V, W = eigen_pairs_divideConqure(G)
        x=0
