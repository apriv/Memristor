from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
#import matplotlib.pyplot as plt
from random import normalvariate
from numpy.linalg import *
from numpy import *
from math import sqrt
import networkx as nx
import numpy as np
import math
import metis
import csv
import time
import metis


class DivideConqure(object):

    def __init__(self):
            self.input=0

    def pre_process2(self,G):
        output = nx.Graph()
        nodes = []
        dict = {}
        counter = 0
        for node in G.nodes():
            nodes.append(counter)
            dict.update({node: counter})
            counter += 1
        output.add_nodes_from(nodes)
        for edge in G.edges():
            output.add_edge(dict[edge[0]], dict[edge[1]])
        return output

    def divide_conqure_old(self, datadir ,graphname, k,ep):
        self.update_parameters(datadir, graphname)
        A = self.inputMatrix
        start = time.time()
        list = []
        Graphs = []
        for p in range(0, k):
            list2 = []
            list.append(list2)
        (edgecuts, parts) = metis.part_graph(self.graph, k)
        for i, p in enumerate(parts):
            list[p].append(self.graph.nodes()[i])
        first = time.time()
        secondTime = 0
        for p in range(0, k):
            H = self.graph.subgraph(list[p])
            A1 = nx.adjacency_matrix(H).toarray()
            eig_val_cov, eig_vec_cov = np.linalg.eig(A1)
            eig_val_cov = sorted(eig_val_cov, reverse=True)
            max1 = eig_val_cov[0]
            min = eig_val_cov[len(eig_val_cov) - 1]
            tempTime1 = time.time()
            #########################################
            v1, w1 = eigen_pairs_computation(A1, len(H.nodes()) / 5, ep, max1, min)
            Graphs.append(w1)
            secondTime = time.time() - tempTime1 + secondTime
        second = time.time()
        third = time.time()
        V0 = self.direct_sum(Graphs)
        fourth = time.time()
        v1, w1 = self.block_lanczos(A, V0, 1, k)
        fifth = time.time()
        v2, w2 = np.linalg.eig(A)
        v2, w2 = self.sort_eigens1(v2, w2)
        remove_duplicates_total(v2, w2)
        similarity1, similarity2 = similarity(v1, v2, w1, w2)
        writer.writerow({"value": similarity1[0][1], "vector": similarity2[0][1]})
        writer.writerow(
            {"1": str(first - start), "2": str(secondTime), "3": str(third - second), "4": str(fourth - third),
             "5": str(fifth - fourth)})

    def divide_conqure2(self, datadir, graphname,ep):
        G = nx.read_edgelist(datadir +"dataset/"+ graphname + ".txt")
        matrix = nx.adjacency_matrix(G)
        G = self.pre_process(G)
        A = matrix.toarray()
        k = 2
        list = []
        Graphs = []
        for p in range(0, k):
            list2 = []
            list.append(list2)
        (edgecuts, parts) = metis.part_graph(G, k)
        for i, p in enumerate(parts):
            list[p].append(G.nodes()[i])
        for p in range(0, k):
            H = G.subgraph(list[p])
            A1 = nx.adjacency_matrix(H).toarray()
            eig_val_cov, eig_vec_cov = np.linalg.eig(A1)
            eig_val_cov = sorted(eig_val_cov, reverse=True)
            max1 = eig_val_cov[0]
            min1 = eig_val_cov[len(eig_val_cov) - 1]
            #########################################
            v1, w1 = self.distinct_eigen_pairs_dynamic(A1, max1, min1, ep,len(H.nodes()))
            Graphs.append(w1)
        V0 = self.direct_sum(Graphs)
        v1, w1 = self.block_lanczos(A, V0, 1, k)
        fifth = time.time()
        v2, w2 = np.linalg.eig(A)
        #v2, w2 = sort_eigens1(v2, w2)
        v2, w2 = self.sort_eigens(v2, w2)
        self.remove_duplicates_total(v2, w2)
        similarity1, similarity2 = self.similarity(v1, v2, w1, w2)

    def find_Distance(self,v):
        min = 10000000
        for i in range(1, len(v)):
            first = v[i - 1]
            second = v[i]
            dist = first - second
            if first != second:
                if abs(dist) < min:
                    min = abs(dist)
        return min

    def direct_sum(self, list):
        item1 = list[0]
        for i in range(1, len(list)):
            item2 = list[i]
            item1 = self.direct_sums(item1, item2)
        return item1

    def direct_sums(self, a, b):
        dsum = np.zeros(np.add(a.shape, b.shape))
        dsum[:a.shape[0], :a.shape[1]] = a
        dsum[a.shape[0]:, a.shape[1]:] = b
        return dsum

    def block_lanczos(self,A, V0, ep, k):
        B0 = self.create_empty_ndarray(len(V0[0]), len(V0[0]))
        Q0 = self.create_empty_ndarray(len(A), len(V0[0]))
        Q1 = V0
        eig_val, eig_vec = np.linalg.eig(A)
        QH = Q1
        k = len(V0[0])
        lastv = []
        lastw = []
        lastd = 100000000
        bool = True
        counter = 0
        while (bool):
            R = np.dot(A, Q1) - np.dot(Q0, B0.transpose())
            A1 = np.dot(Q1.transpose(), R)
            R = R - np.dot(Q1, A1)
            Q2, B1 = np.linalg.qr(R)
            TH1 = np.dot(np.dot(QH.transpose(), A), QH)
            v1, w1 = np.linalg.eig(TH1)
            v1, w1 = self.sort_eigens1(v1, w1)
            w1 = np.dot(QH, w1)
            d = self.check_distance(v1, w1, A, k)
            if d > lastd:
                v1 = lastv
                w1 = lastw
                bool = False
            else:
                counter += 1
                Q0 = Q1
                Q1 = Q2
                QH = np.concatenate((Q2, QH), axis=1)
                B0 = B1
                lastv = v1
                lastw = w1
                lastd = d
        return v1, w1

    def create_empty_ndarray(self,d1, d2):
        output = np.ndarray(shape=(d1, d2), dtype=float, order='F')
        for i in range(d1):
            for j in range(d2):
                output[i][j] = 0.0
        return output

    def check_distance(self,v1, w1, A, k):
        d = 0
        for i in range(k):
            d = d + np.dot(A, w1[:, i]) - np.dot(v1[i], w1[:, i])
        d = d / k
        return self.vec_length(d)

    def vec_length(self,v):
        output = 0
        for item in v:
            output = output + item.real * item.real + item.imag * item.imag
        return sqrt(output.real)

    def sort_eigens1(self,v1, w1):
        w2 = []
        v2 = sorted(v1, reverse=True)
        v3 = []
        for i in range(0, len(v1)):
            v3.append(v2[i])
            x = v2[i]
            b = [item for item in range(len(v1)) if v1[item] == x]
            l = w1[:, b]
            l = l.flatten()
            l = l.tolist()
            w2.append(l)
        w2 = np.matrix(w2)
        w3 = w2.transpose()
        return v3, w3

    def remove_duplicates_total(self,v1, w1):
        w2 = []
        v2 = sorted(v1, reverse=True)
        v3 = []
        for i in range(0, len(v1)):
            if round(v2[i], 10) not in v3:
                v3.append(round(v2[i], 10))
                x = v2[i]
                b = [item for item in range(len(v1)) if v1[item] == x]
                l = w1[:, b]
                l = l.flatten()
                l = l.tolist()
                w2.append(l[0])
        w2 = np.matrix(w2)
        w3 = w2.transpose()
        return v3, w3

        output = []
        for item in input:
            item = round(item, 10)
            if item not in output:
                output.append(item)
        return output

    def eigen_pairs_computation(self,matrix, k, ep, max1, min):
        xs1 = self.eigen_values_computation_perturbation(matrix, k, ep, max1, min)
        eigenvectors = self.eigenvector_computation(matrix, xs1)
        return xs1, eigenvectors

    def eigen_values_computation_perturbation(self, matrix, k, ep, max1, min):
        xs1 = self.eigen_values_computation_revised(matrix, k, ep, max1, min)
        xs2 = self.eigenvalue_computation_duplicates(matrix, k, ep, min, max1, xs1)
        return xs2

    def eigen_values_computation_revised(self,matrix, k, ep, max1, min):
        xs1, ys1, x_vals, y_vals = self.eigen_values_computation(matrix, k, ep, max1, min)
        xs2, ys2 = self.eigenvalue_computation_values(matrix, k, ep / 100, min, xs1)
        xs = []
        for i in range(len(xs2)):
            if ys2[i] > ys1[i] + 10:
                xs.append(xs2[i])
        return xs

    def eigen_values_computation(self,input, k, ep, max1, min):
        from numpy import linalg as LA
        A = np.mat(input)
        b = self.first_n_Primes(len(input))
        # b  = np.random.rand(1, len(input))
        # b= b.flatten()
        # b= b.tolist()
        I = np.identity(len(input))
        x_vals = []
        y_vals = []
        xs = []
        ys = []
        values = []
        values.append(1000)
        values.append(1000)
        eigenvalues = []
        bool = True
        i = 0
        j = 0

        while (bool):
            YI = I * max1
            result = LA.solve(A - YI, b)
            eig = self.find_max(result)
            if j > -1:
                x_vals.append(max1)
                y_vals.append(eig)
            values.append(eig)
            if values[values.index(eig)] == values[values.index(eig) - 1]:
                values.remove(eig)
            elif (values[values.index(eig) - 1] > values[values.index(eig) - 2] and values[values.index(eig) - 1] >
                values[values.index(eig)]) and (
                            (values[values.index(eig) - 1] - values[values.index(eig) - 2]) > 0 and (
                                values[values.index(eig) - 1] - values[values.index(eig)]) > 0):
                eigenvalues.append(round(max1 + ep, 15))
                ys.append(eig)
                xs.append(round(max1 + ep, 15))
                j += 1
            if (len(eigenvalues) >= k or max1 < min - 1):
                bool = False
            i += 1

            max1 -= ep
            if max1 < min - 1:
                eigenvalues.append(0)
                ys.append(0)
                xs.append(round(max1 + ep, 15))
                if (len(eigenvalues) == k):
                    bool = False

        return xs, ys, x_vals, y_vals

    def eigenvalue_computation_values(self,input, k, ep, min, eigens):
        from numpy import linalg as LA
        A = np.mat(input)
        b = self.first_n_Primes(len(input))
        I = np.identity(len(input))
        x_vals = []
        y_vals = []
        values = []
        xs = []
        ys = []
        values.append(1000)
        values.append(1000)
        eigenvalues = []
        bool = True
        i = 0
        j = 0
        # counter =0
        max1 = eigens[j] + (10 * ep)
        if j > 0:
            if eigens[j - 1] >= max1:
                max1 = eigens[j - 1] - ep
        while (bool):
            YI = I * max1
            result = LA.solve(A - YI, b)
            eig = self.find_max(result)
            if j > -1:
                x_vals.append(max1)
                y_vals.append(eig)
            values.append(eig)
            if values[values.index(eig)] == values[values.index(eig) - 1]:
                values.remove(eig)
            elif (values[values.index(eig) - 1] > values[values.index(eig) - 2] and values[values.index(eig) - 1] >
                values[values.index(eig)]) and (
                            (values[values.index(eig) - 1] - values[values.index(eig) - 2]) > 0 and (
                                values[values.index(eig) - 1] - values[values.index(eig)]) > 0):
                eigenvalues.append(round(max1 + ep, 15))
                ys.append(eig)
                xs.append(round(max1 + ep, 15))
                j += 1

                if j < len(eigens):
                    max1 = eigens[j] + (100 * ep)
                    if eigens[j - 1] <= max1:
                        max1 = eigens[j - 1] - ep
            if (len(eigenvalues) >= k or max1 < min - 1):
                bool = False
            i += 1

            max1 -= ep
            if max1 < min - 1:
                eigenvalues.append(0)
                if (len(eigenvalues) == k):
                    bool = False

        return xs, ys

    def find_max(self,input):
        max1 = 0
        for element in input:
            if abs(element) > max1:
                max1 = abs(element)
        return max1

    def first_n_Primes(self,n):
        number_under_test = 4
        primes = [2, 3]
        while len(primes) < n:
            check = False
            for prime in primes:
                if prime > math.sqrt(number_under_test): break
                if number_under_test % prime == 0:
                    check = True
                    break
            if not check:
                for counter in range(primes[len(primes) - 1], number_under_test - 1, 2):
                    if number_under_test % counter == 0:
                        check = True
                        break
            if not check:
                primes.append(number_under_test)
            number_under_test += 1
        return primes

    def eigenvector_computation(self,input, xs):
        eigenvalues = xs
        A = np.mat(input)
        I = np.identity(len(input))
        new_matrix = np.ndarray(shape=(len(A), len(eigenvalues)), dtype=float, order='F')
        for i in range(len(eigenvalues)):
            item = eigenvalues[i].real + 0.1
            vector, x = self.inverse_power(A - item * I)
            new_matrix[:, i] = vector
            i += 1
        return new_matrix

    def similarity(self,v1, v2, w1, w2):
        v3 = []
        similarity2 = 0
        for i in range(len(v1)):
            v3.append(v2[i])
            w3 = w1[:, i]
            w4 = w2[:, i]
            w3 = w3.flatten()
            w4 = w4.flatten()
            w3 = w3.tolist()
            w4 = w4.tolist()
            M1 = np.array([w3[0], w4[0]])
            similarity1 = cosine_similarity(M1)
            similarity2 = similarity2 + similarity1

        similarity2 = similarity2 / len(v1)
        M1 = np.array([v1, v3])
        similarity1 = cosine_similarity(M1)
        return similarity1, similarity2

    def inverse_power(self,E):
        MATRIX_SIZE = len(E)

        rx = np.random.rand(1, MATRIX_SIZE)
        r0 = rx[0]
        w0 = np.linalg.solve(E, r0)
        r = w0 / max(abs(w0))

        w = np.linalg.solve(E, r)

        # Start the inverse_power until convergence
        M = np.array([w0, w])
        # print similarities
        count = 0
        # while (similarities[0][1] < 0.9999999995):
        while (count < 10):
            w0 = w
            w = np.linalg.solve(E, r)
            r = w / max(abs(w))
            count = count + 1
        sum = 0
        for item in w:
            sum += abs(item) * abs(item)
        sum = sqrt(sum)
        w2 = []
        neg = 1
        if w[0] < 0:
            neg = -1
        for item in w:
            w2.append(round(item / sum, 8) * neg)
        return w2, count

    def divide_conqure(self,datadir, graphNames, k):
        for graphname in graphNames:
            with open(datadir + graphname + "newpartition.csv", 'wb') as csvfile:
                obj = EigenPair()
                fieldnames = ["value", "vector", "1", "2", "3", "4", "5"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                G = nx.read_edgelist(datadir + graphname + ".txt")
                obj.pre_process(G)
                G = obj.graph
                matrix = nx.adjacency_matrix(G)
                A = matrix.toarray()
                matrix = np.matrix(A)
                start = time.time()
                list = []
                Graphs = []
                Graphs2 = []

                for p in range(0, k):
                    list2 = []
                    list.append(list2)
                (edgecuts, parts) = metis.part_graph(G, k)
                for i, p in enumerate(parts):
                    list[p].append(G.nodes()[i])
                first = time.time()
                secondTime = 0
                for p in range(0, k):
                    H = G.subgraph(list[p])
                    A1 = nx.adjacency_matrix(H).toarray()
                    eig_val_cov, eig_vec_cov = np.linalg.eig(A1)
                    eig_val_cov = sorted(eig_val_cov, reverse=True)
                    max1 = eig_val_cov[0]
                    min1 = eig_val_cov[len(eig_val_cov) - 1]
                    ep = self.find_Distance(eig_val_cov)
                    ep = 0.001
                    tempTime1 = time.time()
                    #########################################
                    obj = EigenPair()
                    obj.update_parameters_2(A1)
                    obj.eigen_pairs_dynamic(ep)
                    v1 = obj.vals
                    w1 = obj.vecs
                    #v2, w2 = self.eigen_pairs_computation(A1, len(H.nodes()) / 5, ep, max1, min1)
                    w1 = np.array(w1)
                    Graphs.append(w1)
                    #Graphs2.append(w2)
                    secondTime = time.time() - tempTime1 + secondTime
                second = time.time()
                third = time.time()
                V0 = self.direct_sum(Graphs)
                fourth = time.time()
                v1, w1 = self.block_lanczos(A, V0, 1, k)
                fifth = time.time()
                v2, w2 = np.linalg.eig(A)
                v2, w2 = self.sort_eigens1(v2, w2)
                ####
                self.remove_duplicates_total(v2, w2)
                similarity1, similarity2 = self.similarity(v1, v2, w1, w2)
                writer.writerow({"value": similarity1[0][1], "vector": similarity2[0][1]})
                writer.writerow(
                    {"1": str(first - start), "2": str(secondTime), "3": str(third - second), "4": str(fourth - third),
                     "5": str(fifth - fourth)})

    def similarity(self,v1, v2, w1, w2):
        v3 = []
        similarity2 = 0
        for i in range(len(v1)):
            v3.append(v2[i])
            w3 = w1[:, i]
            w4 = w2[:, i]
            w3 = w3.flatten()
            w4 = w4.flatten()
            w3 = w3.tolist()
            w4 = w4.tolist()
            M1 = np.array([w3[0], w4[0]])
            similarity1 = cosine_similarity(M1)
            similarity2 = similarity2 + similarity1

        similarity2 = similarity2 / len(v1)
        M1 = np.array([v1, v3])
        similarity1 = cosine_similarity(M1)
        return similarity1, similarity2

    def eigenvalue_computation_duplicates(self,input, k, ep, min, max1, eigens):
        A = np.mat(input)
        xs1 = eigens
        output = []
        x1 = []
        xx1 = []
        for item in xs1:
            output.append(item)
            x1.append(item)
            xx1.append(item)
        ep2 = 0.02
        while len(output) < k:
            B = np.random.uniform(0, ep2, len(A) * len(A)).reshape(len(A), len(A))
            A = A + B
            matrix = np.matrix(A)
            xs2 = self.eigen_values_computation_revised(matrix, k, ep, max1, min)
            x2 = []
            for item in xs2:
                x2.append(item)
            for item in xs1:
                xs2, y = self.find_closest_eigen(xs2, item, 2)
            for item in xs2:
                xx1, y = self.find_closest_eigen(xx1, item, 2)
                xx1.append(y)
                output.append(y)
                if len(output) == len(A):
                    break

        output = sorted(output, reverse=True)
        return output





if __name__ == '__main__':

    datadir = "/../../dataset/"
    graphNames = ["dolphins", "karate", "dolphins", "lesmis", "squeak", "CA-GrQc", "robots", "fb", "lesmis",
                  "celegansneural", "power", "adjnoun", "polblogs", "polbooks", "netscience", "as-22july06", "G100",
                  "G200", "G500", "G1000", "G2000"]
    obj = DivideConqure()
    obj.divide_conqure(datadir, graphNames, 2)