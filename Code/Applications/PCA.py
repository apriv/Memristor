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






def PCA(G,k, datadir=''):


    matrix = nx.adjacency_matrix(G)
    A = matrix.toarray()

    # 1. Taking the whole dataset ignoring the class labels
    matrix = np.matrix(A)

    # 2. Computing the d-dimensional mean vector
    mean_vector = []
    n = len(G.nodes())
    for i in range(n):
        mean_x = np.mean(matrix[i, :])
        mean_vector.append(mean_x)

    # 3. a) Computing the Scatter Matrix
    scatter_matrix = np.zeros((n, n))
    for i in range(matrix.shape[1]):
        x = matrix[:, i]
        y = matrix[:, i].reshape(n, 1)
        scatter_matrix += (matrix[:, i].reshape(n, 1) - mean_vector).dot(
            (matrix[:, i].reshape(n, 1) - mean_vector).T)

    # 3. b) Computing the Covariance Matrix (alternatively to the scatter matrix)
    cov_list = []
    for i in range(n):
        cov_list.append(A[i, :])
    cov_mat = np.cov(cov_list)
    #####

    # 4. Computing eigenvectors and corresponding eigenvalues
    # eigenvectors and eigenvalues for the from the scatter matrix
    # eig_val_sc, eig_vec_sc = np.linalg.eig(cov_mat)
    #####

    obj = SE.EigenPair()
    obj.update_parameters(cov_mat, isMatrix=True)
    # eigenvectors and eigenvalues for the from the covariance matrix
    eig_val_cov = obj.realVals
    eig_vec_cov = obj.realVecs
    eig_val_sweep, eig_vec_sweep = obj.distinct_eigen_pairs(cov_mat, obj.maximum, obj.minimum, 0.001, isbasic=False, k=k)
    percent, vecError, valError, totalError = obj.evaluate2(0.001, obj.realVals, eig_val_sweep, eig_vec_sweep)

    #####
    # max = eig_val_cov[0]
    # min = eig_val_cov[len(eig_val_cov) - 1]
    # ep = find_Distance(eig_val_cov)
    # eig_val_sweep, eig_vec_sweep, eig_vec_cov = eigen_computation2(cov_mat, k, ep, max, min)

    # eig_val_sweep, eig_vec_sweep = eigen_computation(cov_mat, k, ep, max, min)

    list1 = []
    for i in range(k):
        temp = [x.real for x in obj.realVecs[i]]
        list1.append(temp)
    new_matrix = np.matrix(list1)
    new_matrix = new_matrix.transpose()
    new_matrix2 = np.matrix(eig_vec_sweep)
    new_matrix2 = new_matrix2.transpose()

    eig_vec_sweep = np.array(new_matrix2)
    eig_vec_cov = np.array(new_matrix)
    eig_val_sweep = np.array(eig_val_sweep)
    eig_val_cov = np.array(eig_val_cov)

    # for i in range(len(eig_val_sc)):
    #   eigvec_sc = eig_vec_sc[:, i].reshape(1, n).T
    #  eigvec_cov = eig_vec_cov[:, i].reshape(1, n).T
    # #assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'

    # 5.1. Sorting the eigenvectors by decreasing eigenvalues
    # for ev in eig_vec_sc:
    #    npt.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_sweep))]
    ######
    # eig_pairs2 = [(np.abs(eig_val_sweep[i]), eig_vec_sweep[:, i]) for i in range(len(eig_vec_sweep))]
    # x= np.abs(eig_val_sweep[0])
    # y=eig_vec_sweep2[0]
    eig_pairs3 = [(np.abs(eig_val_sweep[i]), eig_vec_sweep[:, i]) for i in range(len(eig_val_sweep))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    ####3
    # eig_pairs2.sort(key=lambda x: x[0], reverse=True)
    eig_pairs3.sort(key=lambda x: x[0], reverse=True)

    # 5.2. Choosing k eigenvectors with the largest eigenvalues
    tup = ()
    ##
    tup2 = ()
    tup3 = ()
    for i in range(k):
        tup = tup + (eig_pairs[i][1].reshape(n, 1),)
        # tup2 = tup2 + (eig_pairs2[i][1].reshape(n, 1),)
        tup3 = tup3 + (eig_pairs3[i][1].reshape(n, 1),)

    matrix_w = np.hstack(tup)
    ###
    # matrix_w2 = np.hstack(tup2)
    matrix_w3 = np.hstack(tup3)

    # 6. Transforming the samples onto the new subspace
    transformed = matrix_w.T.dot(matrix)
    # transformed2 = matrix_w2.T.dot(matrix)
    transformed3 = matrix_w3.T.dot(matrix)

    ###############
    out1 = 0
    out2 = 0
    out3 = 0
    out4 = 0
    out5 = 0
    out6 = 0
    out7 = 0
    for i in range(0, k - 1):
        M1 = np.array(transformed[i]).flatten()
        # M2=np.array(transformed2[i]).flatten()
        M3 = np.array(transformed3[i]).flatten()
        M1 = M1.tolist()
        M3 = M3.tolist()
        M3 = [item.real for item in M3]
        M = np.array([M1, M3])
        # M2 = np.array([M1, M3])
        #similarities = cosine_similarity(M1, M3)
        similarities = cosine_similarity(M)
        similarities2 = pairwise_distances(M)
        # similarities2 = cosine_similarity(M2)

        similarities3 = manhattan_distances(M)
        similarities4 = euclidean_distances(M)
        out1 += abs(similarities[0][1])
        out2 += abs(similarities2[0][1])
        out3 += abs(similarities3[0][1])
        out4 += abs(similarities4[0][1])
    out1 = out1 / k
    out2 = out2 / k
    out3 = out3 / k
    out4 = out4 / k

    if datadir !='':
        with open(datadir  + "PCA.csv", 'wb') as csvfile:
            fieldnames = ["name", "result"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            writer.writerow({'name': cosine_similarity, 'result': str(out1)})
            writer.writerow({'name': pairwise_distances, 'result': str(out2)})
            writer.writerow({'name': manhattan_distances, 'result': str(out3)})
            writer.writerow({'name': euclidean_distances, 'result': str(out4)})
            writer.writerow({'name': valError, 'result': str(valError)})
            writer.writerow({'name': vecError, 'result': str(vecError)})
            writer.writerow({'name': totalError, 'result': str(totalError)})

    return eig_val_sweep, eig_vec_sweep





if __name__ == '__main__':
    datadir = "../../dataset/"
    graphNames = ["karate", "G100", "G200", "G500", "G1000", "G2000", "G5000", "G10000", "G25000", "G100000"]
    for graphname in graphNames:
        G = nx.read_edgelist(datadir  + graphname + ".txt")
        result = PCA(G, 10)
        x=0

