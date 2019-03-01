from numpy import *
import networkx as nx
import EigenPair




def Eigen(G, ep = 0.001, dir = ""):
    obj = EigenPair.EigenPair()
    obj.update_parameters(G, isDynamic=True)
    obj.eigen_pairs(ep)
    obj.evaluate(ep)
    if dir != "":
        obj.log2(dir, ep)

    return obj.distinctVals, obj.distinctVecs





if __name__ == '__main__':
    datadir = "/Users/sara 1/Documents/PHD/git/Memristor/dataset/"
    graphNames = ["G100", "G200", "G500", "G1000", "G2000", "G5000", "G10000", "G25000", "G100000"]
    for graphname in graphNames:
        print (graphname)
        G = nx.read_edgelist(datadir+ graphname + ".txt")
        V, W = Eigen(G)
        x=0
