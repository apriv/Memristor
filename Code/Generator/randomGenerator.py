from __future__ import division, absolute_import, print_function
import networkx as nx
import sys
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append("/usr/local/lib/python2.7/site-packages")
import csv
import random
import networkx as nx
import itertools
import operator
#import matplotlib.pyplot as plt
from random import randint
import operator
import time
import networkx as nx
# import graphCleanup as gcl
#from pylab import rcParams

#rcParams['figure.figsize'] = 10, 7
#import matplotlib.pyplot as plt
import sys
import csv
import numpy as np
import math
import random
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from numpy import linalg as LA
import scipy.sparse as sparse
from numpy import zeros, dot, identity
#from LUdecomp import *
from math import sqrt
from random import random
#from sympy import Matrix, pretty
import numpy as np
import numpy.testing as npt
import itertools




def random_Graph_Generator(k):
    G = nx.Graph()
    for i in range(k):
        G.add_node(i)
    for i in range(k):
        for j in range(k):
            r = random.randint()
            if r<0.3:
                G.add_edge(i,j)

    return G

def ran_Graph_Generator(datadir):
    count=[4]
    for k in count:
        fname = datadir + "dataset/"+"G" + str(k) + '.txt'
        writer = open(fname, 'w')
        G = nx.Graph()
        for i in range(k):
            G.add_node(i)
        for i in range(k):
            for j in range(k):
                r = np.random.uniform(0, 1)
                if r<0.1:
                    G.add_edge(i,j)
                    writer.write(str(i) + "\t" + str(j) + "\n")
        writer.close()
    return G




if __name__ == '__main__':
    out = []
    datadir = "../../"
    G=ran_Graph_Generator(datadir)



