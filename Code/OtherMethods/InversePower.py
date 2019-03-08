from __future__ import division, absolute_import, print_function
import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
import networkx as nx
from pylab import rcParams
rcParams['figure.figsize'] = 10, 7
import csv
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sparse
from math import sqrt
import numpy as np
from numpy.linalg import inv





def compare_eigenvectors(datadir,graphNames):
    for graphname in graphNames:
        with open(datadir + graphname + "EigenCompare.csv", 'wb') as csvfile:
            fieldnames = ["eigenvalue", "standard","inverse","similarity"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            G = nx.read_edgelist(datadir + "dataset/" + graphname + ".txt")
            matrix = nx.adjacency_matrix(G)
            A = matrix.toarray()
            matrix = np.matrix(A)
           # matrixinv = inv(matrix)
            v, w = np.linalg.eig(matrix)
            I = np.identity(len(v))
            i=0
            list=[]
            v=v.real
            v=v.tolist()
            v2= sorted(v, reverse=True)
            for item in v2:
                x= w[:, v.index(item)]
                if item != 0:
                    invitem= 1/item + 0.1
                else:
                    invitem = 1 / (item + 0.1) + 0.1
                item += 0.1
               # vector, n = inverse_power(matrix - item * I)
                vector, n = inverse_power2(matrix - item * I)

               # invvector, invn = inverse_power(matrixinv - invitem * I)
                x= x.real
                M = np.array([vector, x])
                similarities = cosine_similarity(M)
                #invM = np.array([invvector, x])
               # invsimilarities = cosine_similarity(invM)
                print(round(similarities[0][1],6))
                list.append(vector)
                i+=1
                writer.writerow({"eigenvalue": str(item-0.1), "standard": str(x.tolist()),"inverse":str(vector),"similarity":str(similarities[0][1])})
    return 0





def inverse_power(E):
    MATRIX_SIZE = len(E)
    rx = np.random.rand(1, MATRIX_SIZE)
    r0 = rx[0]
    w0 = np.linalg.solve(E, r0)
    r = w0 / max(abs(w0))
    w = np.linalg.solve(E, r)
    # Start the inverse_power until convergence
    M = np.array([w0, w])
    similarities = cosine_similarity(M)
    count = 0
    #while (similarities[0][1] < 0.9999999995):
    while count<10:
        w0 = w

        w = np.linalg.solve(E, r)
        r = w / max(abs(w))
        M = np.array([w0, w])
        M_sparse = sparse.csr_matrix(M)
        similarities = cosine_similarity(M_sparse)
        count = count + 1
    sum=0
    for item in w:
        sum+=abs(item)*abs(item)
    sum= sqrt(sum)
    w2=[]
    for item in w:
        w2.append(round(item/sum,8))
    return w2, count


def inverse_power2(E):
    MATRIX_SIZE = len(E)
    rx = np.random.rand(1, MATRIX_SIZE)
    r0 = rx[0]
    w0 = np.linalg.solve(E, r0)
    r = w0 / find_norm(w0)
    w = np.linalg.solve(E, r)
    # Start the inverse_power until convergence
    M = np.array([w0, w])
    similarities = cosine_similarity(M)
    count = 0
    #while (similarities[0][1] < 0.9999999995):
    while count<10:
        w0 = w

        w = np.linalg.solve(E, r)
        r = w / max(abs(w))
        M = np.array([w0, w])
        M_sparse = sparse.csr_matrix(M)
        similarities = cosine_similarity(M_sparse)
        count = count + 1
    sum=0
    for item in w:
        sum+=abs(item)*abs(item)
    sum= sqrt(sum)
    w2=[]
    for item in w:
        w2.append(round(item/sum,8))
    return w2, count

def find_norm(v):
    sum=0
    for item in v:
        sum+=item*item
    sum=sqrt(sum)
    return sum














if __name__ == '__main__':
    datadir = "/../"
    graphNames = ["karate","dolphins","squeak","CA-GrQc","fb","robots"]
    compare_eigenvectors(datadir, graphNames)

