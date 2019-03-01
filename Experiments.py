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
import EigenPair



class Cluster(object):
    '''
    A set of points and their centroid
    '''

    def __init__(self, points):
        '''
        points - A list of point objects
        '''

        if len(points) == 0:
            raise Exception("ERROR: empty cluster")

        # The points that belong to this cluster
        self.points = points

        # The dimensionality of the points in this cluster
        #self.n = points[0].n
        self.n = len(points)

        # Assert that all points are of the same dimensionality
        for p in points:
            if p.n != self.n:
                raise Exception("ERROR: inconsistent dimensions")

        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calculateCentroid()

    def __repr__(self):
        '''
        String representation of this object
        '''
        return str(self.points)

    def update(self, points):
        '''
        Returns the distance between the previous centroid and the new after
        recalculating and storing the new centroid.
        Note: Initially we expect centroids to shift around a lot and then
        gradually settle down.
        '''
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid)
        return shift

    def calculateCentroid(self):
        '''
        Finds a virtual center point for a group of n-dimensional points
        '''
        numPoints = len(self.points)
        # Get a list of all coordinates in this cluster
        coords = [p.coords for p in self.points]
        # Reformat that so all x's are together, all y'z etc.
        unzipped = zip(*coords)
        # Calculate the mean for each dimension
        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]

        return Point(centroid_coords)

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

def spectral_Clustering(datadir, graphname, k):
    ep =0.0001
    from scipy.sparse import csgraph
    #for graphname in graphNames:
    with open(datadir + "output/" + graphname + "(" + str(k) + ")" +"outputClustering.csv", 'wb') as csvfile:
        fieldnames = ['adjusted_rand_score', 'adjusted_mutual_info_score', 'homogeneity_score',
                      'v_measure_score', 'fowlkes_mallows_score', 'normalized_mutual_info_score', 'f1_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        obj = EigenPair()
        G = nx.read_edgelist(datadir + "dataset/" + graphname + ".txt")
        matrix = nx.adjacency_matrix(G)
        A = matrix.toarray()
        graph = csgraph.laplacian(A, normed=False)
        A = obj.update_parameters(graph,isMatrix=False)
        # obj.eigen_pairs_dynamic(ep)
        vals, vecs = obj.distinct_eigen_pairs(obj.A, obj.maximum, obj.minimum, ep, k)

        list1 = []
        for i in range(k):
            temp =[x.real for x in obj.realVecs[i]]
            list1.append(temp)
        new_matrix = np.matrix(list1)
        new_matrix = new_matrix.transpose()
        new_matrix2 = np.matrix(vecs)
        new_matrix2 = new_matrix2.transpose()

        FirstCluster = K_means2(new_matrix, k)  # , 0.2)
        FirstClusterrrrr = K_means2(new_matrix, k)  # , 0.2)
        # FirstCluster2 = K_means(eigenVectors11)
        SecondCluster = K_means2(np.array(new_matrix2), k)  # , 0.2)
        result = clustering_evaluation(FirstCluster, SecondCluster)
        # result2 = clustering_evaluation(FirstCluster2, SecondCluster)
        writer.writerow({'adjusted_rand_score': result[0], 'adjusted_mutual_info_score': result[1],
                         'homogeneity_score': result[2], 'v_measure_score': result[3],
                         'fowlkes_mallows_score': result[4], 'normalized_mutual_info_score': result[5],
                         'f1_score': result[6]})

def PCA(datadir,graphname,k):
    with open(datadir + "output/" + graphname+"("+str(k)+")" + "PCA.csv", 'wb') as csvfile:
        fieldnames = ["name", "result"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        G = nx.read_edgelist(datadir + "dataset/" + graphname + ".txt")
        # G = nx.read_gml(datadir+"dataset/gml/" + graphname + ".gml")

        matrix = nx.adjacency_matrix(G)
        A = matrix.toarray()

        # 1. Taking the whole dataset ignoring the class labels
        matrix = np.matrix(A)

        # 2. Computing the d-dimensional mean vector
        mean_vector = []
        n = len(G.nodes())
        for i in range(n):
            mean_x = np.mean(matrix[0, :])
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

        obj = EigenPair()
        obj.update_parameters(cov_mat,isMatrix=True)
        # eigenvectors and eigenvalues for the from the covariance matrix
        eig_val_cov = obj.realVals
        eig_vec_cov = obj.realVecs
        eig_val_sweep, eig_vec_sweep = obj.distinct_eigen_pairs(cov_mat, obj.maximum, obj.minimum, 0.001, k)
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
            M = np.array([M1, M3])
            # M2 = np.array([M1, M3])
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

        writer.writerow({'name': cosine_similarity, 'result': str(out1)})
        writer.writerow({'name': pairwise_distances, 'result': str(out2)})
        writer.writerow({'name': manhattan_distances, 'result': str(out3)})
        writer.writerow({'name': euclidean_distances, 'result': str(out4)})
        writer.writerow({'name': valError, 'result': str(valError)})
        writer.writerow({'name': vecError, 'result': str(vecError)})
        writer.writerow({'name': totalError, 'result': str(totalError)})

def conditionNumber(datadir,graphname):
    with open(datadir + "output/" + graphname + "conditionNumber.csv", 'wb') as csvfile:
        fieldnames = ["name", "result"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        obj = EigenPair()
        G = nx.read_edgelist(datadir + "dataset/" + graphname + ".txt")
        matrix = nx.adjacency_matrix(G)
        A = matrix.toarray()
        B = np.negative(A)
        #graph = csgraph.laplacian(A, normed=False)
        #A =
        obj.update_parameters(A,isMatrix=True)
        # obj.eigen_pairs_dynamic(ep)
        vals1, vecs1= obj.distinct_eigen_pairs(A, obj.maximum, obj.minimum, 0.001, 1)
        vals2, vecs2 = obj.distinct_eigen_pairs(B, -(obj.minimum), -(obj.maximum), 0.001, 1)

        CN = abs(vals1[0]) /abs(vals2[0])
        cnReal = abs(obj.realVals[0]) /abs(obj.realVals[obj.dimension -1])

        error = abs(CN - cnReal)/ abs(cnReal)

        writer.writerow({'name': "cn", 'result': str(CN)})
        writer.writerow({'name': "cnreal", 'result': str(cnReal)})
        writer.writerow({'name': "error", 'result': str(error)})

def pageRank2(datadir,graphname):
    with open(datadir + "output/" + graphname + "page.csv", 'wb') as csvfile:
        fieldnames = ["name", "result"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        obj = EigenPair()
        G = nx.read_edgelist(datadir + "dataset/" + graphname + ".txt")
        obj = EigenPair()
        obj.update_parameters(datadir, graphname)
        start = time.clock()
        #vals, vecs, counters = obj.distinct_eigen_pairs_dynamic_percents(obj.A, 1.2, 0.9, 1,10)
        vals, vecs = obj.distinct_eigen_pairs(obj.A, 1, 0.9, 0.001, 10000)

        result1 = obj.find_closest_eigen2(vals,1)
        pagerank1 = vecs[result1]
        result2 = obj.find_closest_eigen2(obj.realVals, 1)
        pagerank2 = obj.realVecs[result2]
        v1 = pagerank1
        v2 = pagerank2

        error = 0
        c = 0
        for i in range (len(v1)):
            #if abs(v2[i]) > 0.001 :
            error += abs(abs(v1[i]) - abs(v2[i])) / abs(v2[i])
            c+=1
        error = error/c


        M = np.array([v1, v2])
        similarities = cosine_similarity(M)
        similarities2 = pairwise_distances(M)
        similarities3 = manhattan_distances(M)
        similarities4 = euclidean_distances(M)


        writer.writerow({'name': cosine_similarity, 'result': str(similarities[0][1])})
        writer.writerow({'name': pairwise_distances, 'result': str(similarities2[0][1])})
        writer.writerow({'name': manhattan_distances, 'result': str(similarities3[0][1])})
        writer.writerow({'name': euclidean_distances, 'result': str(similarities4[0][1])})
        writer.writerow({'name': error, 'result': str(error)})

def postProcess(input):
    temp=[]
    output=[]
    for val in input:
        if val not in temp:
            temp.append(val)
    for val in input:
        index=temp.index(val)
        output.append(index)
    return output

def K_means2(input,k):
    from sklearn.cluster import KMeans
    kmeans = KMeans(k)
    output= kmeans.fit_predict(input)
    kmeans = KMeans(n_clusters=k, random_state=1).fit(input)
    output= kmeans.labels_
    output=postProcess(output)
    return output

def kmeans(points, k, cutoff):

    # Pick out k random points to use as our initial centroids
    initial = random.sample(points, k)

    # Create k clusters using those centroids
    # Note: Cluster takes lists, so we wrap each point in a list here.
    clusters = [Cluster([p]) for p in initial]

    # Loop through the dataset until the clusters stabilize
    loopCounter = 0
    while True:
        # Create a list of lists to hold the points in each cluster
        lists = [[] for _ in clusters]
        clusterCount = len(clusters)

        # Start counting loops
        loopCounter += 1
        # For every point in the dataset ...
        for p in points:
            # Get the distance between that point and the centroid of the first
            # cluster.
            smallest_distance = getDistance(p, clusters[0].centroid)

            # Set the cluster this point belongs to
            clusterIndex = 0

            # For the remainder of the clusters ...
            for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's
                # centroid.
                distance = getDistance(p, clusters[i+1].centroid)
                # If it's closer to that cluster's centroid update what we
                # think the smallest distance is
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i+1
            # After finding the cluster the smallest distance away
            # set the point to belong to that cluster
            lists[clusterIndex].append(p)

        # Set our biggest_shift to zero for this iteration
        biggest_shift = 0.0

        # For each cluster ...
        for i in range(clusterCount):
            # Calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            biggest_shift = max(biggest_shift, shift)

        # If the centroids have stopped moving much, say we're done!
        if biggest_shift < cutoff:
            print "Converged after %s iterations" % loopCounter
            break
    return clusters

def clustering_evaluation(First,Second):
    output=[]
    from sklearn import metrics
    output.append( metrics.adjusted_rand_score(First, Second))
    output.append(metrics.adjusted_mutual_info_score(First, Second))
    output.append(metrics.homogeneity_score(First, Second))
    output.append(metrics.v_measure_score(First, Second))
    output.append(metrics.fowlkes_mallows_score(First, Second))
    output.append(metrics.normalized_mutual_info_score(First, Second))
    output.append(metrics.f1_score(First, Second,average='micro'))
    return output

def experimentRandom():
    ep =0.0001
    datadir = "/Users/sara 1/Documents/PHD/Python/"
    graphNames = ["G100","G500","G1000","G2000","G5000","G10000"]
    for graphname in graphNames:
        obj = EigenPair()
        obj.update_parameters(datadir, graphname)
        obj.eigen_pairs(ep)
        obj.logRound(datadir, graphname, ep)


    print ("finish")

def experimentRandom2():
    sizes=[100,500,1000,2000,5000,10000]
    for size in sizes:
        print(str(size))
        print("***")
        obj = EigenPair()
        ep = 0.0001
        datadir = "/Users/sara 1/Documents/PHD/Python/"
        #f = open(datadir  +"output/"+ "random " + str(size) + ".", 'w')
        A = np.random.uniform(-1, 1, size * size).reshape(size, size)
        matrix = np.matrix(A)
        obj.update_parameters(matrix,isMatrix=True)
        obj.eigen_pairs(ep)
        # obj.evaluate(ep)
        obj.logRound2(datadir, str(size), ep)


    print ("finish")

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
            obj.evaluate(ep)
            obj.log2(datadir, graphname, ep)

    print ("finish")

def experiment2(datadir, graphNames):
    eps = [0.001]
    for graphname in graphNames:
        print ("****************graphname:  " + graphname)
        for ep in eps:
            print ("****************ep  :" + str(ep))
            obj = EigenPair()
            G = nx.read_edgelist(datadir + "dataset/" + graphname + ".txt")
            obj.update_parameters(G)
            obj.eigen_pairs(ep)
            obj.evaluate(ep)
            obj.log2(datadir, graphname, ep)

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

def fivepercentresults(datadir, graphNames):
    ep = 0.001
    percent = 5
    for graphname in graphNames:
        print ("****************graphname:  " + graphname)
       # print ("****************ep  :" + str(ep))
        obj = EigenPair()
        obj.update_parameters(datadir, graphname)
        obj.eigen_pairs_dynamic_percents(percent)
        obj.log_distinct_Percents(datadir, graphname, ep, percent)

    print ("finish")

def pageRank(G, s = .85, maxerr = .0001):
    n = G.shape[0]
    # transform G into markov matrix A
    A = csc_matrix(G,dtype=np.float)
    rsums = np.array(A.sum(1))[:,0]
    ri, ci = A.nonzero()
    A.data /= rsums[ri]
    # bool array of sink states
    sink = rsums==0
    #    Compute pagerank r until we converge
    ro, r = np.zeros(n), np.ones(n)
    while np.sum(np.abs(r-ro)) > maxerr:
        ro = r.copy()
# calculate each pagerank at a time
    for i in xrange(0,n):
# inlinks of state i
        Ai = np.array(A[:,i].todense())[:,0]
# account for sink states
        Di = sink / float(n)
# account for teleportation to state i
        Ei = np.ones(n) / float(n)
        r[i] = ro.dot( Ai*s + Di*s + Ei*(1-s) )
# return normalized pagerank
    return r/float(sum(r))
#if __name__=='__main__':
# Example extracted from 'Introduction to Information Retrieval'
 #   G = np.array([[0,0,1,0,0,0,0],
     #             [0,1,1,0,0,0,0],
      #            [1,0,1,1,0,0,0],
       #           [0,0,0,1,1,0,0],
        #          [0,0,0,0,0,0,1],
         #         [0,0,0,0,0,1,1],
          #        [0,0,0,1,1,0,1]])
#print pageRank(G,s=.86)












if __name__ == '__main__':
    #experiment6()

    datadir = "/Users/sara 1/Documents/PHD/Python/"
    graphNames = ["karate","G100", "G200", "G500","G1000","G2000","G5000","G10000","G25000", "G100000"]
    experiment8(datadir,graphNames)
    for graphname in graphNames:
        print (str(graphname))
        pageRank2(datadir, graphname)


