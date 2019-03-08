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
#import EigenPair
import sys
sys.path.insert(0, '../SweepingMethod/')
import EigenPair as SE
from scipy.sparse import csgraph

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
            print("Converged after %s iterations" % loopCounter)
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

def spectral_Clustering(G, k, datadir=''):
    ep =0.0001
    #for graphname in graphNames:

    obj = SE.EigenPair()
    #G = nx.read_edgelist(datadir + "dataset/" + graphname + ".txt")
    matrix = nx.adjacency_matrix(G)
    A = matrix.toarray()
    graph = csgraph.laplacian(A, normed=False)
    obj.update_parameters(graph, isMatrix=True)
    # obj.eigen_pairs_dynamic(ep)
    vals, vecs = obj.distinct_eigen_pairs(obj.A, obj.maximum, obj.minimum, ep, isbasic=False, k= k)

    list1 = []
    for i in range(k):
        temp = [x.real for x in obj.realVecs[i]]
        list1.append(temp)
    new_matrix = np.matrix(list1)
    new_matrix = new_matrix.transpose()
    new_matrix2 = np.matrix(vecs)
    new_matrix2 = new_matrix2.transpose()



    if datadir !='':
        with open(datadir + "outputClustering.csv", 'wb') as csvfile:
            fieldnames = ['adjusted_rand_score', 'adjusted_mutual_info_score', 'homogeneity_score',
                          'v_measure_score', 'fowlkes_mallows_score', 'normalized_mutual_info_score', 'f1_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

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








if __name__ == '__main__':
    #experiment
    datadir = "../../dataset/"
    graphNames = ["karate", "G100", "G200", "G500", "G1000", "G2000", "G5000", "G10000", "G25000", "G100000"]
    for graphname in graphNames:
        G = nx.read_edgelist(datadir + graphname + ".txt")
        spectral_Clustering(G, 10)
