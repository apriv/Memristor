from numpy import *
import networkx as nx
import EigenPair
import time



#Basic version (ISVLSI paper)
def EigenBasic(G, ep = 0.001, dir =""):
    obj = EigenPair.EigenPair()
    obj.update_parameters(G, isDynamic=False)
    obj.eigen_pairs(ep, isbasic=True)
    obj.evaluate(ep)
    if dir != "":
        obj.log2(dir, ep)

    return obj.distinctVals, obj.distinctVecs


#Static
def EigenStatic(G, ep = 0.001, dir = ""):
    obj = EigenPair.EigenPair()
    obj.update_parameters(G, isDynamic=False)
    obj.eigen_pairs(ep)
    obj.evaluate(ep)
    if dir != "":
        obj.log2(dir, ep)

    return obj.distinctVals, obj.distinctVecs


#Dynamic
def Eigen(G, ep = 0.001, dir = ""):
    obj = EigenPair.EigenPair()
    obj.update_parameters(G, isDynamic=True)
    obj.eigen_pairs(ep)
    obj.evaluate(ep)
    if dir != "":
        obj.log2(dir, ep)

    return obj.distinctVals, obj.distinctVecs


#dynamic with repeating eigens
def EigenTotal(G, ep = 0.001, dir = ""):
    obj = EigenPair.EigenPair()
    obj.update_parameters(G, isDynamic=True)
    obj.eigen_pairs(ep)
    obj.evaluate(ep)
    obj.repeating_eigen_pairs(ep)
    if dir != "":
        obj.log2(dir, ep)

    return obj.vals, obj.vecs


#asymmetric
def EigenAsym():
    obj = EigenPair()
    ep = 0.0001
    A = np.random.uniform(-1, 1, size * size).reshape(size, size)
    matrix = np.matrix(A)
    obj.update_parameters(matrix, isMatrix=True, isSymmetric=False, isDynamic=True)
    obj.eigen_pairs(ep)


if __name__ == '__main__':
    datadir = "../../dataset/"
    graphNames = [ "karate","G100", "G200", "G500", "G1000", "G2000", "G5000", "G10000", "G25000", "G100000"]
    graphNames = [ "karate","G100", "G200", "G500", "G1000", "G2000", "G5000"]

    for graphname in graphNames:
        time_start = time.time()
        print (graphname)
        G = nx.read_edgelist(datadir+ graphname + ".txt")
        V, W = Eigen(G)
        x=0
        time_end = time.time()
        print('time cost', time_end - time_start, 's')

