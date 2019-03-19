from numpy import *
import networkx as nx
import EigenPair
import time



def Eigen(G, ep = 0.001, dir = ""):
    obj = EigenPair.EigenPair()
    obj.update_parameters(G, isDynamic=True)
    obj.eigen_pairs(ep)
    obj.evaluate(ep)
    if dir != "":
        obj.log2(dir, ep)

    return obj.distinctVals, obj.distinctVecs





if __name__ == '__main__':
    datadir = "../dataset/"
    graphNames = ["G100", "G200", "G500", "G1000"]
    for graphname in graphNames:
        print (graphname)
        G = nx.read_edgelist(datadir+ graphname + ".txt")
        before = time.time()
        V, W = Eigen(G)
        after = time.time()
        print(before-after)
        x=0
