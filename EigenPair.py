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


class EigenPair(object):

    ## defining parameters
    def __init__(self):
        self.dimension = 0
        self.maximum = 0
        self.minimum = 0
        self.eps = []
        self.realVals = []
        self.realVecs = []
        self.distinctVals = []
        self.distinctVecs = []
        self.repeatingVals = []
        self.vals = []
        self.vecs = []
        self.inputMatrix = []
        self.A = []
        self.b1 = []
        self.b2 = []
        self.I = []
        self.percent = 0
        self.valError = 0
        self.vecError = 0
        self.totalError = 0
        self.percents = {}
        self.valErrors = {}
        self.vecErrors = {}
        self.totalErrors = {}
        self.percentsQR = []
        self.valErrorsQR = []
        self.vecErrorsQR = []
        self.totalErrorsQR = []
        self.timesQR =[]
        self.time = 0
        self.graph = []
        self.counter =0
        self.isSymmetric = True
        self.isDynamic = True
        self.maxreal = 0
        self.maximag = 0
        self.minreal = 0
        self.minimag = 0

    #
        # initialization

    ##initialization
    def update_parameters(self, G, isSymmetric=True, isMatrix=False, isDynamic=True):
        self.isDynamic = isDynamic
        self.isSymmetric = isSymmetric
        if isMatrix:
            self.A = G
            self.inputMatrix = np.matrix(self.A)
        else:
            self.graph = self.pre_process(G)
            matrix = nx.adjacency_matrix(G)
            self.A = matrix.toarray()
            self.inputMatrix = np.matrix(self.A)
        self.eigen_space = self.eigenspace(self.A)
        v, w = np.linalg.eig(self.inputMatrix)
        self.realVals, self.realVecs = self.sort_eigens(v, w)
        self.dimension = len(self.realVals)
        self.maximum = self.realVals[0]
        self.minimum = self.realVals[-1]
        self.b1 = self.first_n_Primes(len(self.realVals))
        self.b2 = self.first_n_Primes(len(self.realVals))
        random.shuffle(self.b2)
        self.b3 = self.perpendicular(self.b2)
        self.b4 = self.first_n_Primes(len(self.realVals))
        self.b5 = self.first_n_Primes(len(self.realVals))
        random.shuffle(self.b4)
        random.shuffle(self.b5)
        self.I = np.identity(len(self.realVals))
        if isSymmetric == False:
            Acomplex = []
            b1complex = []
            b2complex = []
            b3complex = []
            b4complex = []
            b5complex = []
            Icomplex = []
            for i in range(len(self.A)):
                Acomplex.append([])
                Icomplex.append([])
                b1complex.append(self.b1[i] + 1j * (0))
                b2complex.append(self.b2[i] + 1j * (0))
                b3complex.append(self.b3[i] + 1j * (0))
                b4complex.append(self.b4[i] + 1j * (0))
                b5complex.append(self.b5[i] + 1j * (0))
                for j in range(len(self.A)):
                    Acomplex[i].append(self.A[i][j] + 1j * (0))
                    Icomplex[i].append(self.I[i][j] + 1j * (0))
            self.A = Acomplex
            self.inputMatrix = np.matrix(self.A)
            self.b1 = b1complex
            self.b2 = b2complex
            self.b3 = b3complex
            self.b4 = b4complex
            self.b5 = b5complex
            self.I = np.identity(len(self.realVals))

            values = np.array(self.realVals)
            reals = list(values.real)
            imags = sorted(list(values.imag), reverse=True)
            self.maxreal = reals[0]
            self.minreal = reals[-1]
            self.maximag = imags[0]
            self.minimag = -imags[0]

    ## set the nodes to a list of numbers in order from zero
    def pre_process(self, G):
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
        self.graph = output

    ## calculating the eigen space interval
    def eigenspace(self, G):
        intervals = []
        A = np.array(G)
        for i in range(len(A)):
            point = A[i][i]
            p = 0
            for j in range(len(A)):
                if i != j:
                    p += abs(A[i][j])
            intervals.append((point - p, point + p, False))
        intervals = set(intervals)
        intervals = list(intervals)
        intervals.sort(key=lambda tup: tup[0])

        result = [intervals[0]]
        for i in xrange(1, len(intervals)):
            prev, current = result[-1], intervals[i]
            if current[0] <= prev[1]:
                result[-1] = (prev[0], max(prev[1], current[1]), True)
            else:
                result.append(current)
        return result

    ##sort calculared eigens
    def sort_eigens(self, v1, w1):
        dic = dict(zip(v1, range(len(v1))))
        v1 = sorted(v1, reverse=True)
        w2=[]
        for i in xrange(len(v1)):
            l = w1[:, dic[v1[i]]]
            l = l[:, 0]
            l = l.flatten()
            l = l.tolist()
            l = l[0]
            w2.append(l)
        return v1, w2

    ## generating a vector of co-prime numbers
    def first_n_Primes(self, n):
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

    ## generate a perpendicular vector to vec
    def perpendicular(self, vec):
        output = []
        l = len(vec) / 2
        for i in range(l):
            x1 = vec[2 * i]
            x2 = vec[2 * i + 1]
            output.append(x2)
            output.append(-x1)
        if len(output) < len(vec):
            output.append(0)
        return output

    ## generate eigenpairs of any kind
    def eigen_pairs(self, ep):
        start = time.clock()
        if self.isSymmetric:
            vals, vecs = self.distinct_eigen_pairs(self.A, self.maximum, self.minimum, ep)
        else:
            vals, vecs = self.distinct_eigen_pairs_asym2( ep)
        self.distinctVals = vals
        self.distinctVecs = vecs
        #self.repeating_eigen_pairs(ep)
        self.time = time.clock() - start

    ##find max absolute element
    def find_max(self, input):
        return abs(max(input, key=abs))

    ## find just distinct eigenn pairs
    def eigen_pairs_Distinct(self, ep, k=-1):
        start = time.clock()
        vals, vecs = self.distinct_eigen_pairs(self.A, self.maximum, self.minimum, ep,k)
        self.vals = vals
        self.vecs = vecs
        self.time = time.clock() - start

    ## find distinct eigenn pairs
    def distinct_eigen_pairs(self, A, max1, min1, ep, k=-1):
        if k ==-1: k = self.dimension
        max1 += 0.5
        eigenvalues = []
        eigenvectors = []
        isEigen = False
        counter = 1
        value = max1
        r2 = -1
        r3 = -1
        result3=[]
        while (True):
            counter += 1
            max1 = value
            r1 = r2
            r2 = r3
            result2 = result3
            result3 = LA.solve(A - self.I * (max1 - (ep)), self.b3)
            r3 = self.find_max(result3)
            value = max1 - (ep)
            if ((r2 > r3 and r2 > r1)):
                isEigen = True
                flag, result21 = self.verifyEigens(A, max1, ep)
                if flag:
                    eigval, eigvec = self.compute_eigenvec(result2, result21, A, max1)
                    eigenvalues.append(eigval)
                    eigenvectors.append(eigvec)
            if isEigen:
                isEigen = False
            if max1 < min1 - 1 or len(eigenvalues) >= k:
                break
            if self.isDynamic: ep = self.compute_delta(r1, r2, r3, ep)
        return eigenvalues, eigenvectors

    # find distinct eigenn pairs asym
    def distinct_eigen_pairs_ASym(self, A, maxreal, minreal, maximag, minimag, ep, k=-1):
        if k == -1: k = self.dimension
        eigenvalues = []
        eigenvectors = []
        isEigen = False
        counter = 1
        valuereal = maxreal
        r2 = -1
        r3 = -1
        result3 = []
        flagg = True
        while (maxreal > minreal and flagg):
            maxreal = valuereal
            valuereal = maxreal - ep
            valueimag = maximag
            maximag1 =  maximag
            while (maximag1 > minimag and flagg):
                counter += 1
                maximag1 = valueimag
                valueimag = maximag1 - ep
                r1 = r2
                r2 = r3
                result2 = result3
                comp = maxreal + 1j * (maximag1)
                temp = np.multiply(self.I, comp)
                A = self.A - temp
                result3 = result3 = np.linalg.solve(A, self.b3)
                r3 = self.find_max_complex(result3)
                if ((r2 > r3 and r2 > r1)):
                    isEigen = True
                    flag, result21 = self.verifyComplex(self.A, self.I, maxreal - ep, maximag1 -ep, ep)
                    if flag:
                        eigval, eigvec = self.compute_eigenvec_asym(result2, result21, A, maxreal - ep, maximag1 -ep)
                        eigenvalues.append(eigval)
                        eigenvectors.append(eigvec)
                if isEigen:
                    isEigen = False
                if len(eigenvalues) >= k:
                    flagg = False
                #if self.isDynamic: ep = self.compute_delta(r1, r2, r3, ep)
        return eigenvalues, eigenvectors

    def distinct_eigen_pairs_asym2(self, ep, isDynamic=True):
        k = self.dimension
        ep = 0.01
        #
        mmax1 = 0
        for item in self.realVals:
            mmax1 = max(mmax1, item.imag)
        mmax2 =  self.realVals[0].real
        #
        for item in self.eigen_space:
            min1 = item[0]
            max1 = item[1]
            #
            max1 = mmax2
            mmax1 = mmax1
            #
            mincomp =-item[2]
            #flag = item[3]
            values=[max1]
            r3 = -1
            r2 = -1
            eigenvalues = []
            eigenvectors = []
            isEigen = False
            result3 = []
            ##
            flag2 = True
            while (max1>min1 and flag2):
                max1 = values[-1]
                realpart = max1 - ep
                values.append(max1 - (ep))
                #maxcomp = item[2]
                #
                maxcomp = mmax1
                #
                valuescomp = [maxcomp]
                while (maxcomp>mincomp and flag2):
                    r1 = r2
                    r2 = r3
                    result2 = result3
                    maxcomp =  valuescomp[-1]
                    imagpart = maxcomp - ep
                    comp = realpart + 1j * (imagpart)
                    temp = np.multiply(self.I, comp)
                    A = self.A - temp
                    result3 = np.linalg.solve(A, self.b3)
                    r3 = self.find_max_complex(result3)
                    valuescomp.append(maxcomp - (ep))
                    if ((r2 > r3 and r2 > r1)):
                        isEigen = True
                        flag, result21 = self.verifyComplex(self.A, self.I, max1, maxcomp, ep)
                        if flag:

                            eigval, eigvec = self.compute_eigenvec_asym(result2, result21, A, max1,maxcomp)
                            #eigenvalues.append(eigval)
                            #eigenvectors.append(eigvec)
                            print(str((max1,maxcomp)))
                            eigenvalues.append(eigval)
                            eigenvectors.append(eigvec)
                    if isEigen:
                        isEigen = False
                    #if isDynamic: ep = self.compute_delta(r1, r2, r3, ep)

                if max1 < min1 - 1 or len(eigenvalues) >= k:
                    flag2 = False

        self.vals = eigenvalues
        self.vecs = eigenvectors

        return eigenvalues, eigenvectors


    def verifyComplex(self, Acomplex,Icomplex, maxreal,maxComplex, ep):
        flag = False
        '''
        comp = (maxreal+ 2 * ep + 1j) * (maxComplex)
        temp = np.multiply(Icomplex, comp)
        result10 = np.linalg.solve(Acomplex - temp, self.b5)
        r01 = self.find_max_complex(result10)

        ##
        comp = (maxreal + 1j) * (maxComplex+ 2 * ep )
        temp = np.multiply(Icomplex, comp)
        result10 = np.linalg.solve(Acomplex - temp, self.b5)
        r01c = self.find_max_complex(result10)
        ##
        '''

        comp = (maxreal +  ep) + (1j * (maxComplex))
        temp = np.multiply(Icomplex, comp)
        result20 = np.linalg.solve(Acomplex - temp, self.b5)
        #result20 = self.eigenvectester(Acomplex - temp, self.b5)
        r11 = self.find_max_complex(result20)

        ##
        comp = maxreal + (1j * (maxComplex+ ep))
        temp = np.multiply(Icomplex, comp)
        result20 = np.linalg.solve(Acomplex - temp, self.b5)
        r11c = self.find_max_complex(result20)
        ##
        comp = maxreal + (1j * (maxComplex))
        temp = np.multiply(Icomplex, comp)
        result30 = np.linalg.solve(Acomplex - temp, self.b5)
        r21 = self.find_max_complex(result30)

        comp = (maxreal - ep) + (1j * (maxComplex))
        temp = np.multiply(Icomplex, comp)
        result40 = np.linalg.solve(Acomplex - temp, self.b5)
        r31 = self.find_max_complex(result40)

        ##
        comp = (maxreal)  + (1j * (maxComplex- ep))
        temp = np.multiply(Icomplex, comp)
        result40 = np.linalg.solve(Acomplex - temp, self.b5)
        r31c = self.find_max_complex(result40)
        ##
        '''
        comp = (maxreal- 2 * ep + 1j) * (maxComplex)
        temp = np.multiply(Icomplex, comp)
        result50 = np.linalg.solve(Acomplex - temp, self.b5)
        r41 = self.find_max_complex(result50)

        ##
        comp = (maxreal + 1j) * (maxComplex- 2 * ep)
        temp = np.multiply(Icomplex, comp)
        result50 = np.linalg.solve(Acomplex - temp, self.b5)
        r41c = self.find_max_complex(result50)
        ##

        if r11 > r01 and r11 > r21:
            r31 = r21
            r21 = r11
            result20 = result10
            r11 = r01
        elif r31 > r21 and r31 > r41:
            r11 = r21
            r21 = r31
            result20 = result30
            r31 = r41
        '''
        if (r21 > r11) and (r21 > r31) and (r21 > r11c) and (r21 > r31c):
            flag = True

        return flag, result30


    def find_max_complex(self, input):
        max1total =0
        for element in input:
            x = abs(element.imag)
            y = (x * x)
            z = abs(element.real)
            a = (z*z)
            aa = y + a
            max1total += aa
        return  max1total

    ## find repeating eigenpairs
    def repeating_eigen_pairs(self, ep):
        self.eigenvalue_computation_duplicates(ep)
        self.eigenvector_computation()

    ## verifying detected eigenvalues
    def verifyEigens(self, A, max1, ep):
        flag = False
        result10 = LA.solve(A - self.I * (max1 + 2 * ep), self.b5)
        r01 = self.find_max(result10)
        result11 = LA.solve(A - self.I * (max1 + ep), self.b5)
        r11 = self.find_max(result11)
        result21 = LA.solve(A - self.I * (max1), self.b5)
        r21 = self.find_max(result21)
        result31 = LA.solve(A - self.I * (max1 - ep), self.b5)
        r31 = self.find_max(result31)
        result41 = LA.solve(A - self.I * (max1 - 2 * ep), self.b5)
        r41 = self.find_max(result41)
        if r11 > r01 and r11 > r21:
            r31 = r21
            r21 = r11
            result21 = result11
            r11 = r01
        elif r31 > r21 and r31 > r41:
            r11 = r21
            r21 = r31
            result21 = result31
            r31 = r41
        if (r21 > r11) and (r21 > r31):
            flag = True
        return flag, result21

    ## compute eigenvector corresponding to detected eigenvalues
    def compute_eigenvec(self, result2, result21, A, max1):
        resultx1 = self.normalize(result2)
        resultx2 = self.normalize(result21)
        a1 = np.dot(A, resultx1)
        a1 = a1.tolist()
        a1 = a1[0]
        a2 = np.dot(max1, resultx1)
        a3 = a1 - a2
        a3 = [round(item, 3) for item in a3]
        maximum1 = max(a3)
        a1 = np.dot(A, resultx2)
        a1 = a1.tolist()
        a1 = a1[0]
        a2 = np.dot(max1, resultx2)
        a3 = a1 - a2
        a3 = [round(item, 3) for item in a3]
        maximum2 = max(a3)
        if maximum2 < maximum1:
            eigvec = resultx2
        else:
            eigvec = resultx1
        eigval = round(max1.real, 15)
        return eigval, eigvec

    def compute_eigenvec_asym2(self, result2, result21, A, max1, maxcomp):
        eigvec = []
        eigval = -1
        #resultx1 = result2
        #resultx2 = result21
        resultx1 = self.normalize_complex_arr(result2)
        resultx2 = self.normalize_complex_arr(result21)
        a1 = np.dot(A, resultx1)
        a1 = a1.tolist()
        # a1 = a1[0]
        a2 = np.dot(max1 + (1j * maxcomp), resultx1)
        a3 = a1 - a2
        maximum1 = max(a3)
        # a4 = [round(item, 3) for item in a3]
        a3 = [round(item.real, 3) + 1j * round(item.imag, 3) for item in a3]

        a1 = np.dot(A, resultx2)
        a1 = a1.tolist()
        # a1 = a1[0]
        a2 = np.dot(max1 + (1j * maxcomp), resultx2)
        a3 = a1 - a2
        maximum2 = max(a3)
        # a4 = [round(item, 3) for item in a3]
        a3 = [round(item.real, 3) + 1j * round(item.imag, 3) for item in a3]

        if maximum2 < maximum1:
            eigvec = resultx2
        else:
            eigvec = resultx1
        eigval = round(max1, 15) + (1j * round(maxcomp, 15))
        return eigval, eigvec

    def compute_eigenvec_asym(self, result2, result21, A, max1, maxcomp):
        eigvec = []
        eigval = -1
        #resultx1 = result2
        #resultx2 = result21
        resultx1 = self.normalize_complex_arr(result2)
        resultx2 = self.normalize_complex_arr(result21)
        a1 = np.dot(self.A, resultx1)
        a1 = a1.tolist()
        # a1 = a1[0]
        a2 = np.dot(max1 + (1j * maxcomp), resultx1)
        a3 = a1 - a2
        maximum1 = max(a3)
        # a4 = [round(item, 3) for item in a3]
        a3 = [round(item.real, 3) + 1j * round(item.imag, 3) for item in a3]

        a1 = np.dot(self.A, resultx2)
        a1 = a1.tolist()
        # a1 = a1[0]
        a2 = np.dot(max1 + (1j * maxcomp), resultx2)
        a3 = a1 - a2
        maximum2 = max(a3)
        # a4 = [round(item, 3) for item in a3]
        a3 = [round(item.real, 3) + 1j * round(item.imag, 3) for item in a3]

        if maximum2 < maximum1:
            eigvec = resultx2
        else:
            eigvec = resultx1
        eigval = round(max1, 15) + (1j * round(maxcomp, 15))
        return eigval, eigvec



    def normalize_complex_arr(self, a):
        a_oo = a.real
        a_oo2 = a.imag  # origin offsetted
        vmag = self.magnitude(a_oo)
        vmag2 = self.magnitude(a_oo2)
        #m1  = max(vmag)
        #m2 = max(vmag2)
        m3 = max(vmag,vmag2)
        vmag = vmag + vmag2
        vmag = m3
        return [a[i] / vmag for i in range(len(a))]

       # return a_oo / np.abs(a_oo).max()

    def normalize_complex_arr_back(self, a):
        a_oo = a - a.real.min() - 1j * a.imag.min()  # origin offsetted
        return a_oo / np.abs(a_oo).max()

    ## compute step size
    def compute_delta(self, r1, r2, r3, ep):
        min1 = 0.001
        max1 = 0.1
        s1 = abs(float(r2 - r1) / ep)
        s2 = abs(float(r2 - r3) / ep)
        s3 = s1 / s2
        d = max(s2, s3)
        if d == 0:
            d = 0.1
        ep = 1 / d
        ep = min(max1, ep)
        ep = max(min1, ep)
        return ep

    #one eigen???
    def eigen_val(self, ep, max1):
        eigvec = []
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

        # max1 = self.maximum
        while (bool):
            YI = self.I * max1
            result = LA.solve(self.A - YI, self.b1)

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

                result1 = LA.solve(self.A - self.I * (max1 + ep + ep), self.b2)
                r1 = self.find_max(result1)
                result2 = LA.solve(self.A - self.I * (max1 + ep), self.b2)
                r2 = self.find_max(result2)
                result3 = LA.solve(self.A - self.I * (max1), self.b2)
                r3 = self.find_max(result3)
                if (r2 > r1) and (r2 > r3):
                    resultx1 = self.normalize(result2)
                    resultx2 = self.normalize(LA.solve(self.A - self.I * (max1 + ep), self.b1))
                    a1 = np.dot(self.A, resultx1)
                    a1 = a1.tolist()
                    a1 = a1[0]
                    a2 = np.dot(max1 + ep, resultx1)
                    a3 = a1 - a2
                    a3 = [round(item, 3) for item in a3]
                    maximum1 = max(a3)
                    a1 = np.dot(self.A, resultx2)
                    a1 = a1.tolist()
                    a1 = a1[0]
                    a2 = np.dot(max1 + ep, resultx2)
                    a3 = a1 - a2
                    a3 = [round(item, 3) for item in a3]
                    maximum2 = max(a3)

                    if maximum2 < maximum1:
                        eigvec.append(resultx2)
                    else:
                        eigvec.append(resultx1)

                    eigenvalues.append(round(max1.real + ep, 15))
                    ys.append(values[values.index(eig) - 1])
                    xs.append(round(max1.real + ep, 15))
                    j += 1

            if (len(eigenvalues) >= 1):
                bool = False
            i += 1

            max1 -= ep

        # self.distinctVals =xs
        # self.distinctVecs = eigvec
        return xs

    ## repeating eigenvalue
    def eigenvalue_computation_duplicates(self, ep):
        dis_vals = self.distinctVals
        output = []
        vals1 = []
        vals2 = []
        for item in dis_vals:
            output.append(item)
            vals1.append(item)
            vals2.append(item)
        ep2 = 10 * ep
        counter = 0
        A = self.A
        while len(output) < self.dimension:
            counter += 1
            B = np.random.uniform(0, ep2, self.dimension * self.dimension).reshape(self.dimension, self.dimension)
            A = A + B
            matrix = np.matrix(A)
            vals, vecs = self.distinct_eigen_pairs(matrix, self.maximum, self.minimum, ep)
            if len(vals) > len(output):
                for item in dis_vals:
                    vals, y = self.find_closest_eigen(vals, item, 2)
                for item in vals:
                    vals2, y = self.find_closest_eigen(vals2, item, 2)
                    if y != -10000:
                        vals2.append(y)
                        output.append(y)
                    if len(output) == len(A):
                        break
                dis_vals = []
                for item in output:
                    dis_vals.append(item)
                break
            else:
                if counter >= 5:
                    break
        output = sorted(output, reverse=True)
        self.repeatingVals = output

    ## detecting real eigenvalue close to this one
    def find_closest_eigen(self, valList, val, ep):
        if valList[0] < val: output = valList[0]
        elif valList[-1] > val: output = valList[-1]
        else:
            for i in range(len(valList) - 1):
                if valList[i] > val and valList[i + 1] < val:
                    if abs(valList[i] - val) > abs(valList[i+1] - val):
                        output = valList[i+1]
                    else: output = valList[i]
                    break
        dist = abs(output - val)
        if dist < 0.5:
            valList.remove(output)
        else:
            output = -10000
        return valList, output




    #???
    def find_closest_eigen2(self, valList, val):
        counter = 0
        min = 1000
        for item in valList:
            dist = abs(item - val)
            if dist < min:
                output = item
                min = dist
        counter = 0
        for item in valList:
            if item == output:
                return counter
            else:
                counter = counter+1


        return counter


    def null(self, A, eps=1e-3):
        u, s, vh = svd(A, full_matrices=1, compute_uv=1)
        null_space = compress(s <= eps, vh, axis=0)
        vector = np.array(null_space)
        vector2 = vector.flatten()
        x = vector2.tolist()
        y = len(x)
        vh = np.array(vh)
        null_spacetest = vh[len(vh) - 1, :]
        if y == 0:
            vh = np.array(vh)
            null_space = vh[len(vh) - 1, :]
            null_space = np.matrix(null_space)
        return null_space.T

    ## vector finctions....
    def normalize(self, v):
        vmag = self.magnitude(v)
        return [v[i] / vmag for i in range(len(v))]
    ##
    def magnitude(self, v):
        return math.sqrt(sum(v[i] * v[i] for i in range(len(v))))

    ##
    def add(self, u, v):
        return [u[i] + v[i] for i in range(len(u))]

    ##
    def sub(self, u, v):
        return [u[i] - v[i] for i in range(len(u))]

    ##
    def dot(self, u, v):
        return sum(u[i] * v[i] for i in range(len(u)))

    ##
    def matVecMul(self, A, X):
        A = np.array(A)
        X = np.array(X)
        result = (np.dot(A, X)).tolist()
        return result

    ##
    def vec_dist(self, V1, V2):
        result = -1
        dist = 0
        if len(V1) == len(V2):
            for i in range(len(V1)):
                dist += (V1[i] - V2[i]) * (V1[i] - V2[i])
            result = sqrt(dist.real)
        return result

    def eigenvector_computation(self):
        vectors = []
        values = []
        eigenvalues = self.repeatingVals
        i = 0
        k = 0
        lens = len(eigenvalues)
        v1 = self.distinctVals
        w1 = self.distinctVecs

        """
        for i in range(len(v1)):
            vv = v1[i].real
            vector = self.null(self.A - vv * self.I)
            vector = np.array(vector)
            vector2 = vector.flatten()
            x = vector2.tolist()
            y = len(x)
            while (y == 0):
                item = item + 0.0001
                vector = self.null(self.A - (item) * self.I)
                vector = np.array(vector)
                vector2 = vector.flatten()
                x = vector2.tolist()
                y = len(x)
            l = len(vector[0, :])
            ww = w1[i]
            result4 = LA.solve(self.A - vv * self.I, ww)

            if l==1:
                M1 = np.array([x, ww])
                similarity1 = cosine_similarity(M1)
                c=0
            else:
                www = self.inverse_power(self.A - (vv + 0.1) * self.I)
                for ii in range(l):
                    M1 = np.array([vector[:, ii].tolist(), ww])
                    M2 = np.array([vector[:, ii].tolist(), www[0]])
                    M3 = np.array([ww, www[0]])


                    a= np.dot(self.A - vv * self.I, ww)
                    similarity1 = cosine_similarity(M1)
                    similarity2 = cosine_similarity(M2)
                    similarity3 = cosine_similarity(M3)

                    c = 0

        """


        ll = self.dimension
        while (True):
            if (eigenvalues[i].real not in values):
                item = eigenvalues[i].real + 0.0001
                if i < len(eigenvalues) - 1 and (eigenvalues[i].real == eigenvalues[i + 1].real):
                    if i > 0:
                        max = eigenvalues[i].real + 0.46545
                        if max > eigenvalues[i - 1].real:
                            max = eigenvalues[i - 1] - 0.004763535
                        xs1 = self.eigen_val(0.0001, max)
                        item = xs1[0].real
                    vector = self.null(self.A - item * self.I)
                    vector = np.array(vector)
                    vector2 = vector.flatten()
                    x = vector2.tolist()
                    y = len(x)
                    while (y == 0):
                        item = item + 0.0001
                        vector = self.null(self.A - (item) * self.I)
                        vector = np.array(vector)
                        vector2 = vector.flatten()
                        x = vector2.tolist()
                        y = len(x)
                    l = len(vector[0, :])
                    temppppp1 =[]
                    temppppp2 =[]
                    temppppp1.append((vector[:, 0].tolist()))
                    for j in range(l):
                        vectors.append(vector[:, j].tolist())
                       ##  bb= vector[:, j].tolist()
                        bb = temppppp1[j]

                        result3 = LA.solve(self.A - item * self.I, bb)
                        temppppp2.append(vector[:, j].tolist())
                        temppppp1.append(result3)

                        values.append(eigenvalues[i].real)
                        k += 1
                        lens += (l - 1)
                    temppppp1.append(LA.solve(self.A - item * self.I, result3))

                    for i in range(len(temppppp1)):
                        for j in range(len(temppppp2)):
                            M1 = np.array([temppppp1[i], temppppp2[j]])
                            similarity1 = cosine_similarity(M1)
                            if similarity1[0][1]>0.9:
                                x = j




                else:
                    if self.repeatingVals[i] == -10000:
                        x = 0
                    ind = v1.index(self.repeatingVals[i])
                    if ind == -10000:
                        x = 0
                    vector, x = w1[ind], v1[ind]
                    vectors.append(vector)
                    values.append(eigenvalues[i].real)
                    k += 1
            i += 1
            if i >= len(eigenvalues) or k >= ll:
                break

        self.vals = values
        self.vecs = vectors

    def evaluate(self, ep):
        correctList = []
        for item in self.realVals:
            correctList.append(item)
        correct = 0
        valDist = 0
        vecDist = 0

        for i in range(len(self.vals)):
            appVal = self.vals[i]
            appVec = self.vecs[i]

            correctList, value = self.find_closest_eigen(correctList, appVal, ep)
            if value != -10000 and value != 10000:
                correct += 1
                vvv = abs((value - appVal) / appVal)
                if (vvv > 2 and abs(appVal) > 0 and abs(appVal) < 1):
                    valDist += 0.5
                else:
                    valDist += abs((value - appVal) / appVal)

                dist = self.matVecMul(self.A - self.I * value, appVec)
                vecDist += self.vec_dist(dist, [0] * self.dimension)

        self.percent = float(correct) / float(self.dimension)
        self.vecError = vecDist / self.dimension
        self.valError = valDist / self.dimension
        self.totalError = (self.valError + self.vecError) / 2

    def evaluateASymmetric(self, ep):
        correctList = []
        for item in self.realVals:
            correctList.append(item)
        correct = 0
        valDist = 0
        vecDist = 0

        for i in range(len(self.distinctVals)):
            appVal = self.vals[i]
            appVec = self.vecs[i]
            #appVec = self.realVecs[i]

            correctList, value = self.find_closest_eigen(correctList, appVal, ep)
            if value != -10000 and value != 10000:
                correct += 1
                vvv = abs((value - appVal) / appVal)
                if (vvv > 2 and abs(appVal) > 0 and abs(appVal) < 1):
                    valDist += 0.5
                else:
                    valDist += abs((value - appVal) / appVal)

                dist = self.matVecMul(self.A - self.I * value, appVec)
                vecDist += self.vec_dist2(dist, [0] * self.dimension)

        self.percent = float(correct) / float(self.dimension)
        self.vecError = vecDist / self.dimension
        self.valError = valDist / self.dimension
        self.totalError = (self.valError + self.vecError) / 2


    def vec_dist2(self, V1, V2):
        result = -1
        dist = 0
        if len(V1) == len(V2):
            for i in range(len(V1)):
                dist += (V1[i] - V2[i]) * (V1[i] - V2[i])
            a1 = dist.imag
            a2 = dist.real
            result = sqrt(abs(dist.real))
            result2 = sqrt(abs(dist.imag))
        return result

    def evaluatePercents(self, ep):
        percent = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        j = 0
        for item in percent:
            realVals = []
            appVals = []
            appVecs = []
            l = (self.dimension * item) / 100
            for i in range(len(self.vals)):
                appVals.append(self.vals[i])
                appVecs.append(self.vecs[i])
            for i in range(len(self.realVals)):
                realVals.append(self.realVals[i])
            per, vecError, valError, totalError = self.evaluate2(ep, realVals, appVals, appVecs)
            self.percents.update({str(j): per})
            self.vecErrors.update({str(j): vecError})
            self.valErrors.update({str(j): valError})
            self.totalErrors.update({str(j): totalError})
            j += 1

    def evaluatePercentsDistinct(self, ep):
        percent = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        j = 0
        for item in percent:
            realVals = []
            appVals = []
            appVecs = []
            l = (self.dimension * item) / 100
            for i in range(l):
                if i < len(self.vals):
                    appVals.append(self.vals[i])
                    appVecs.append(self.vecs[i])
            for i in range(len(self.realVals[i])):
                if self.realVals[i] not in realVals:
                    realVals.append(self.realVals[i])
            per, vecError, valError, totalError = self.evaluate2(ep, realVals, appVals, appVecs)
            self.percents.update({str(j): per})
            self.vecErrors.update({str(j): vecError})
            self.valErrors.update({str(j): valError})
            self.totalErrors.update({str(j): totalError})
            j += 1

    def evaluate2(self, ep, realVal, appVals, appVecs):
        correctList = []
        for item in realVal:
            correctList.append(item)
        correct = 0
        valDist = 0
        vecDist = 0

        for i in range(len(appVals)):
            appVal = appVals[i]
            appVec = appVecs[i]

            correctList, value = self.find_closest_eigen(correctList, appVal, ep)
            if value != -10000:
                correct += 1
                valDist += abs((value - appVal) / appVal)

                dist = self.matVecMul(self.A - self.I * value, appVec)
                vecDist += self.vec_dist(dist, [0] * self.dimension)

        percent = float(correct) / float(len(realVal))
        vecError = vecDist / self.dimension
        valError = valDist / self.dimension
        totalError = (valError + vecError) / 2
        return percent, vecError, valError, totalError



    def log2(self, datadir, graphname, ep):
        self.evaluatePercents(ep)
        with open(datadir + "output/" + graphname + "(" + str(ep) + ")" + "resultdynamiceps.csv", 'wb') as csvfile:
            fieldnames = ["name"]
            for i in range(self.dimension):
                fieldnames.append(str(i))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            dict0 = {"name": "realval"}
            dict1 = {"name": "realvec"}
            dict2 = {"name": "appval"}
            dict3 = {"name": "appvec"}
            dict4 = {"name": "percents"}
            dict5 = {"name": "valErros"}
            dict6 = {"name": "vecErrors"}
            dict7 = {"name": "totalErrors"}
            dict8 = {"name": "time"}

            for i in range(len(self.realVals)):
                dict0.update({str(i): self.realVals[i]})
                dict1.update({str(i): self.realVecs[i]})
            for i in range(len(self.vals)):
                dict2.update({str(i): self.vals[i]})
                dict3.update({str(i): self.vecs[i]})
            dict4.update(self.percents)
            dict5.update(self.valErrors)
            dict6.update(self.vecErrors)
            dict7.update(self.totalErrors)
            dict8.update({'0': self.time})
            writer.writerow(dict0)
            writer.writerow(dict1)
            writer.writerow(dict2)
            writer.writerow(dict3)
            writer.writerow(
                {"name": "percents", '0': "5", '1': "10", '2': "15", '3': "20", '4': "25", '5': "30", '6': "35",
                 '7': "40", '8': "45", '9': "50", '10': "55", '11': "60", '12': "65", '13': "70", '14': "75",
                 '15': "80", '16': "85", '17': "90", '18': "95", '19': "100"})
            writer.writerow(dict4)
            writer.writerow(dict5)
            writer.writerow(dict6)
            writer.writerow(dict7)
            writer.writerow(dict8)

    def log_distinct(self, datadir, graphname, ep):
        self.evaluatePercents(ep)
        with open(datadir + "output/" + graphname + "(" + str(ep) + ")" + "resultdynamiceps.csv", 'wb') as csvfile:
            fieldnames = ["name"]
            for i in range(self.dimension):
                fieldnames.append(str(i))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            dict0 = {"name": "realval"}
            dict1 = {"name": "realvec"}
            dict2 = {"name": "appval"}
            dict3 = {"name": "appvec"}
            dict4 = {"name": "percents"}
            dict5 = {"name": "valErros"}
            dict6 = {"name": "vecErrors"}
            dict7 = {"name": "totalErrors"}
            dict8 = {"name": "time"}

            for i in range(len(self.realVals)):
                dict0.update({str(i): self.realVals[i]})
                dict1.update({str(i): self.realVecs[i]})
            for i in range(len(self.vals)):
                dict2.update({str(i): self.vals[i]})
                dict3.update({str(i): self.vecs[i]})
            dict4.update(self.percents)
            dict5.update(self.valErrors)
            dict6.update(self.vecErrors)
            dict7.update(self.totalErrors)
            dict8.update({'0': self.time})
            writer.writerow(dict0)
            writer.writerow(dict1)
            writer.writerow(dict2)
            writer.writerow(dict3)
            writer.writerow(
                {"name": "percents", '0': "5", '1': "10", '2': "15", '3': "20", '4': "25", '5': "30", '6': "35",
                 '7': "40", '8': "45", '9': "50", '10': "55", '11': "60", '12': "65", '13': "70", '14': "75",
                 '15': "80", '16': "85", '17': "90", '18': "95", '19': "100"})
            writer.writerow(dict4)
            writer.writerow(dict5)
            writer.writerow(dict6)
            writer.writerow(dict7)
            writer.writerow(dict8)

    def log_distinct_Percents(self, datadir, graphname, ep, percent):
        k = (len(self.A) * percent)/100
        with open(datadir + "output/" + graphname + "(" + str(ep) + ")" + "resultdynamiceps.csv", 'wb') as csvfile:
            fieldnames = ["name"]
            for i in range(self.dimension):
                fieldnames.append(str(i))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            dict0 = {"name": "realval"}
            dict1 = {"name": "realvec"}
            dict2 = {"name": "appval"}
            dict3 = {"name": "appvec"}
            dict4 = {"name": "percents"}
            dict5 = {"name": "valErros"}
            dict6 = {"name": "vecErrors"}
            dict7 = {"name": "totalErrors"}
            dict8 = {"name": "time"}
            dict9 = {"name": "counts"}

            realVal = []
            appVals = []
            appVecs = []

            for i in range(k):
                dict0.update({str(i): self.realVals[i]})
                realVal.append(self.realVals[i])
                dict1.update({str(i): self.realVecs[i]})
            for i in range(len(self.Vals)):
                dict2.update({str(i): self.Vals[i]})
                appVals.append(self.Vals[i])
                dict3.update({str(i): self.Vecs[i]})
                appVecs.append(self.Vecs[i])

            percent, vecError, valError, totalError = self.evaluate2(ep, realVal, appVals, appVecs)


            dict4.update({'0' : percent})
            dict5.update({'0' : valError})
            dict6.update({'0' : vecError})
            dict7.update({'0' : totalError})
            dict8.update({'0': self.time})
            dict9.update({'0': self.counter})

            writer.writerow(dict0)
            writer.writerow(dict1)
            writer.writerow(dict2)
            writer.writerow(dict3)
            writer.writerow(
                {"name": "percents", '0': "5", '1': "10", '2': "15", '3': "20", '4': "25", '5': "30", '6': "35",
                 '7': "40", '8': "45", '9': "50", '10': "55", '11': "60", '12': "65", '13': "70", '14': "75",
                 '15': "80", '16': "85", '17': "90", '18': "95", '19': "100"})
            writer.writerow(dict4)
            writer.writerow(dict5)
            writer.writerow(dict6)
            writer.writerow(dict7)
            writer.writerow(dict8)
            writer.writerow(dict9)



    def logRound2(self, datadir, graphname, ep):
        self.evaluatePercents(ep)
        mean1 = mean(self.eps)
        median1 = median(self.eps)
        std1 = std(self.eps)

        with open(datadir + "output/" + graphname + "Round2" + ".csv", 'wb') as csvfile:
            fieldnames = ["name"]
            for i in range(self.dimension):
                fieldnames.append(str(i))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            dict0 = {"name": "realval"}
            dict1 = {"name": "realvec"}
            dict2 = {"name": "appval"}
            dict3 = {"name": "appvec"}
            dict4 = {"name": "percents"}
            dict5 = {"name": "valErros"}
            dict6 = {"name": "vecErrors"}
            dict7 = {"name": "totalErrors"}
            dict8 = {"name": "time"}
            dict9 = {"name": "mean"}
            dict10 = {"name": "median"}
            dict11 = {"name": "std"}


            for i in range(len(self.realVals)):
                dict0.update({str(i): self.realVals[i]})
                dict1.update({str(i): self.realVecs[i]})
            for i in range(len(self.vals)):
                dict2.update({str(i): self.vals[i]})
                dict3.update({str(i): self.vecs[i]})
            dict4.update(self.percents)
            dict5.update(self.valErrors)
            dict6.update(self.vecErrors)
            dict7.update(self.totalErrors)
            dict8.update({'0': self.time})
            dict9.update({'0': mean1})
            dict10.update({'0': median1})
            dict11.update({'0': std1})


            writer.writerow(dict0)
            writer.writerow(dict1)
            writer.writerow(dict2)
            writer.writerow(dict3)
            writer.writerow(
                {"name": "percents", '0': "5", '1': "10", '2': "15", '3': "20", '4': "25", '5': "30", '6': "35",
                 '7': "40", '8': "45", '9': "50", '10': "55", '11': "60", '12': "65", '13': "70", '14': "75",
                 '15': "80", '16': "85", '17': "90", '18': "95", '19': "100"})
            writer.writerow(dict4)
            writer.writerow(dict5)
            writer.writerow(dict6)
            writer.writerow(dict7)
            writer.writerow(dict8)
            writer.writerow(dict9)
            writer.writerow(dict10)
            writer.writerow(dict11)

    def logRound(self, datadir, graphname, ep):
        self.evaluatePercents(ep)
        mean1 = mean(self.eps)
        median1 = median(self.eps)
        std1 = std(self.eps)

        with open(datadir + "output/" + graphname + "Round" + ".csv", 'wb') as csvfile:
            fieldnames = ["name"]
            for i in range(self.dimension):
                fieldnames.append(str(i))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            dict0 = {"name": "realval"}
            dict1 = {"name": "realvec"}
            dict2 = {"name": "appval"}
            dict3 = {"name": "appvec"}
            dict4 = {"name": "percents"}
            dict5 = {"name": "valErros"}
            dict6 = {"name": "vecErrors"}
            dict7 = {"name": "totalErrors"}
            dict8 = {"name": "time"}
            dict9 = {"name": "mean"}
            dict10 = {"name": "median"}
            dict11 = {"name": "std"}


            for i in range(len(self.realVals)):
                dict0.update({str(i): self.realVals[i]})
                dict1.update({str(i): self.realVecs[i]})
            for i in range(len(self.vals)):
                dict2.update({str(i): self.vals[i]})
                dict3.update({str(i): self.vecs[i]})
            dict4.update(self.percents)
            dict5.update(self.valErrors)
            dict6.update(self.vecErrors)
            dict7.update(self.totalErrors)
            dict8.update({'0': self.time})
            dict9.update({'0': mean1})
            dict10.update({'0': median1})
            dict11.update({'0': std1})


            writer.writerow(dict0)
            writer.writerow(dict1)
            writer.writerow(dict2)
            writer.writerow(dict3)
            writer.writerow(
                {"name": "percents", '0': "5", '1': "10", '2': "15", '3': "20", '4': "25", '5': "30", '6': "35",
                 '7': "40", '8': "45", '9': "50", '10': "55", '11': "60", '12': "65", '13': "70", '14': "75",
                 '15': "80", '16': "85", '17': "90", '18': "95", '19': "100"})
            writer.writerow(dict4)
            writer.writerow(dict5)
            writer.writerow(dict6)
            writer.writerow(dict7)
            writer.writerow(dict8)
            writer.writerow(dict9)
            writer.writerow(dict10)
            writer.writerow(dict11)


    def log(self, datadir, graphname, ep):
        self.evaluatePercents(ep)
        with open(datadir + "output/" + graphname + "(" + str(ep) + ")" + "result.csv", 'wb') as csvfile:
            fieldnames = ["name"]
            for i in range(self.dimension):
                fieldnames.append(str(i))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            dict0 = {"name": "realval"}
            dict1 = {"name": "realvec"}
            dict2 = {"name": "appval"}
            dict3 = {"name": "appvec"}
            dict4 = {"name": "percents"}
            dict5 = {"name": "valErros"}
            dict6 = {"name": "vecErrors"}
            dict7 = {"name": "totalErrors"}
            dict8 = {"name": "time"}

            for i in range(len(self.realVals)):
                dict0.update({str(i): self.realVals[i]})
                dict1.update({str(i): self.realVecs[i]})
            for i in range(len(self.vals)):
                dict2.update({str(i): self.vals[i]})
                dict3.update({str(i): self.vecs[i]})
            dict4.update(self.percents)
            dict5.update(self.valErrors)
            dict6.update(self.vecErrors)
            dict7.update(self.totalErrors)
            dict8.update({'0': self.time})
            writer.writerow(dict0)
            writer.writerow(dict1)
            writer.writerow(dict2)
            writer.writerow(dict3)
            writer.writerow(
                {"name": "percents", '0': "5", '1': "10", '2': "15", '3': "20", '4': "25", '5': "30", '6': "35",
                 '7': "40", '8': "45", '9': "50", '10': "55", '11': "60", '12': "65", '13': "70", '14': "75",
                 '15': "80", '16': "85", '17': "90", '18': "95", '19': "100"})
            writer.writerow(dict4)
            writer.writerow(dict5)
            writer.writerow(dict6)
            writer.writerow(dict7)
            writer.writerow(dict8)


    def vector_vector(self, u, v):
        output = np.random.uniform(0, 10, len(u) * len(v)).reshape(len(u), len(v))
        for i in range(len(u)):
            for j in range(len(v)):
                output[i][j] = float(float(u[i]) * float(v[j]))
        return output

    def vec_length(self, v):
        out = 0
        for item in v:
            out += item * item
        out = sqrt(out)

        return out

    def vec_norm(self, v):
        out = 0
        for item in v:
            out += abs(item)
        return out

    def vec_norm2(self, v):
        out = 0
        for item in v:
            out += abs(item * item)
        return sqrt(out)

    def vect_arrowhead(self, D, z, m):
        w = []
        for i in range(len(D)):
            if (D[i] - m) != 0:
                w.append(z[i] / (D[i] - m))
            else:
                w.append(z[i] / (D[i] - (m + 0.001)))
        w.append(-1)
        x = self.vec_norm2(w)
        if x == 0:
            x = 0.001
        w = [item / x for item in w]
        return w

    def householder_transformation(self, a):
        n = len(a)
        for k in range(0, n - 2):
            b = a
            s = 0
            for i in range(k + 1, n):
                s += b[i][k] * a[i][k]
            s = sqrt(s)
            sg = 0
            if s > 0:
                if b[k + 1][k] < 0:
                    sg = -1
                else:
                    sg = 1
            z = float(float((float(1 + (sg * b[k + 1][k]) / s))) / 2.0)
            v = []
            for i in range(n):
                v.append(0)
            v[k + 1] = sqrt(z)
            for i in range(k + 2, n):
                v[i] = float(float(sg) * float(b[k][i])) / float(float(2) * float(v[k + 1]) * float(s))
            H = self.vector_vector(v, v)
            H = np.identity(len(H)) - 2 * H
            x = 0
            a = np.dot(H, b)
            a = np.dot(a, H)
            for i in range(n):
                for j in range(n):
                    a[i][j] = round(a[i][j], 3)
        return a

    def householder_transformation(self, a):
        n = len(a)
        for k in range(0, n - 2):
            b = a
            s = 0
            for i in range(k + 1, n):
                s += b[i][k] * a[i][k]
            s = sqrt(s)
            sg = 0
            if s > 0:
                if b[k + 1][k] < 0:
                    sg = -1
                else:
                    sg = 1
            z = float(float((float(1 + (sg * b[k + 1][k]) / s))) / 2.0)
            v = []
            for i in range(n):
                v.append(0)
            v[k + 1] = sqrt(z)
            for i in range(k + 2, n):
                v[i] = float(float(sg) * float(b[k][i])) / float(float(2) * float(v[k + 1]) * float(s))
            H = self.vector_vector(v, v)
            H = np.identity(len(H)) - 2 * H
            x = 0
            a = np.dot(H, b)
            a = np.dot(a, H)
            for i in range(n):
                for j in range(n):
                    a[i][j] = round(a[i][j], 3)
        return a

    def householder_transformation2(self, a):
        n = len(a)
        P = np.identity(len(a))
        for k in range(0, n - 2):
            b = a
            s = 0
            for i in range(k + 1, n):
                s += b[i][k] * a[i][k]
            s = sqrt(s)
            sg = 0
            if s > 0:
                if b[k + 1][k] < 0:
                    sg = -1
                else:
                    sg = 1
            z = float(float((float(1 + (sg * b[k + 1][k]) / s))) / 2.0)
            v = []
            for i in range(n):
                v.append(0)
            v[k + 1] = sqrt(z)
            for i in range(k + 2, n):
                v[i] = float(float(sg) * float(b[k][i])) / float(float(2) * float(v[k + 1]) * float(s))
            H = self.vector_vector(v, v)
            H = np.identity(len(H)) - 2 * H
            PH = np.zeros(shape=(len(H), len(H)))
            for i in range(len(PH)):
                for j in range(len(PH)):
                    PH[i][j] = round(H[i][j], 3)
            H = PH
            x = 0
            a = np.dot(H, b)
            a = np.dot(a, H)
            P = np.dot(H, P)
            for i in range(n):
                for j in range(n):
                    P[i][j] = round(P[i][j], 3)

        return a, P

    def householder_transformation3(self, a):
        n = len(a)
        aaa = np.zeros(shape=(n, n))
        ccc = np.zeros(shape=(n, n))
        ddd = np.zeros(shape=(n, n))

        for i in range(n):
            for j in range(n):
                aaa[i][j] = a[i][j]
                ccc[i][j] = a[i][j]
                ddd[i][j] = a[i][j]

        P = np.identity(len(a))
        for k in range(0, n - 2):
            b = a
            s = 0
            for i in range(k + 1, n):
                s += b[i][k] * a[i][k]
            s = sqrt(s)
            sg = 0
            if s > 0:
                if b[k + 1][k] < 0:
                    sg = -1
                else:
                    sg = 1
            z = float(float((float(1 + (sg * b[k + 1][k]) / s))) / 2.0)
            v = []
            for i in range(n):
                v.append(0)
            v[k + 1] = sqrt(z)
            for i in range(k + 2, n):
                v[i] = float(float(sg) * float(b[k][i])) / float(float(2) * float(v[k + 1]) * float(s))
            H = self.vector_vector(v, v)
            H = np.identity(len(H)) - 2 * H
            PH = np.zeros(shape=(len(H), len(H)))
            for i in range(len(PH)):
                for j in range(len(PH)):
                    PH[i][j] = round(H[i][j], 3)
            H = PH
            x = 0
            a = np.dot(H, b)
            a = np.dot(a, H)
            P = np.dot(H, P)
            # for i in range(n):
            #    for j in range(n):
            #        P[i][j] = round(P[i][j], 3)
            aaa = np.dot(H, aaa)
            aaa = np.dot(aaa, H)

        PP = np.linalg.inv(np.matrix(P))
        PP = np.array(PP)

        """
        aaaaaa = np.dot(P, ccc)
        aaaaaa = np.dot(aaaaaa, np.transpose(P))

        PP = np.linalg.inv(np.matrix(P))
        PP = np.array(PP)

        dddd = np.dot(PP, a)
        dddd = np.dot(dddd, np.transpose(PP))
        dddd = np.array(dddd)

        for i in range(len(dddd)):
            M1 = np.array([dddd[i], ddd[i]])
            similarity1 = cosine_similarity(M1)
            x = 0

        ####
        vv, ww = np.linalg.eig(np.matrix(ddd))
        vv, ww = self.sort_eigens(vv, ww)
        vv2, ww2 = np.linalg.eig(np.matrix(a))
        vv2, ww2 = self.sort_eigens(vv2, ww2)

        ######
        ws = np.dot(PP, np.transpose(ww2))
        ws = np.transpose(ws)
        ws = ws.tolist()

        for i in range(len(dddd)):
            M1 = np.array([ws[i], ww[i]])
            similarity1 = cosine_similarity(M1)
            x = 0
        """

        return a, PP

    def QR_transformation(self):
        start = time.clock()
        eigvals=[]
        eigvecs=[]
        a = self.A
        n = len(a)
        counter = 0
        counters =[10,20,30,40,50,100]#,3000,4000,5000,6000,7000,8000,9000,10000]
        aaa = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n):
                aaa[i][j] = a[i][j]

        P = np.identity(len(a))
        Pinv = np.identity(len(a))
        while (counter<101):
            counter+=1
            q, r = np.linalg.qr(a)
            q1, r1 = self.householder_reflection(a)
            x = 0
            a = np.dot(np.array(np.linalg.inv(np.matrix(q))), a)
            a = np.dot(a, q)
            P = np.dot(np.array(np.linalg.inv(np.matrix(q))), P)
            Pinv = np.dot(Pinv, q)

            aa = np.dot(r,q)
            xx= a[0][n-1]
            if counter in counters :
                ws = Pinv
                vs = []
                for i in range(n):
                    vs.append(a[i][i])
                vs, ws = self.sort_eigens(vs, np.matrix(ws))
                eigvals.append(vs)
                eigvecs.append(ws)
                self.timesQR.append(time.clock() - start)

        return eigvals,eigvecs

    def inverse_power(self, E):
        MATRIX_SIZE = len(E)

        # rx = np.random.rand(1, MATRIX_SIZE)
        r0 = self.first_n_Primes2(MATRIX_SIZE)
        # r0 = rx[0]
        w0 = np.linalg.solve(E, r0)
        r = w0 / max(abs(w0))

        w = np.linalg.solve(E, r)

        # Start the inverse_power until convergence
        M = np.array([w0, w])
        M_sparse = sparse.csr_matrix(M)
        similarities = cosine_similarity(M)
        # print similari ties

        count = 0
        while (similarities[0][1] < 0.9999999995):
            w0 = w

            w = np.linalg.solve(E, r)
            r = w / max(abs(w))
            M = np.array([w0, w])
            M_sparse = sparse.csr_matrix(M)
            similarities = cosine_similarity(M_sparse)
            count = count + 1
        w2 = []
        for item in w:
            w2.append(round(item, 5))

        return w2, count

    def first_n_Primes2(self,n):
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

    def householder_transformation4(self, a):
        n = len(a)
        aaa = np.zeros(shape=(n, n))
        ccc = np.zeros(shape=(n, n))
        ddd = np.zeros(shape=(n, n))

        for i in range(n):
            for j in range(n):
                aaa[i][j] = a[i][j]
                ccc[i][j] = a[i][j]
                ddd[i][j] = a[i][j]

        P = np.identity(len(a))
        Pinv = np.identity(len(a))
        for k in range(0, n - 2):
            b = a
            s = 0
            for i in range(k + 1, n):
                s += b[i][k] * a[i][k]
            s = sqrt(s)
            sg = 0
            if s > 0:
                if b[k + 1][k] < 0:
                    sg = -1
                else:
                    sg = 1
            z = float(float((float(1 + (sg * b[k + 1][k]) / s))) / 2.0)
            v = []
            for i in range(n):
                v.append(0)
            v[k + 1] = sqrt(z)
            for i in range(k + 2, n):
                v[i] = float(float(sg) * float(b[k][i])) / float(float(2) * float(v[k + 1]) * float(s))
            H = self.vector_vector(v, v)
            H = np.identity(len(H)) - 2 * H
            PH = np.zeros(shape=(len(H), len(H)))
            for i in range(len(PH)):
                for j in range(len(PH)):
                    PH[i][j] = round(H[i][j], 3)
            H = PH
            x = 0
            a = np.dot(H, b)
            a = np.dot(a, H)
            P = np.dot(H, P)
            Pinv = np.dot(Pinv, H)

            #for i in range(n):
             #  for j in range(n):
              #     P[i][j] = round(P[i][j], 3)
            aaa = np.dot(H, aaa)
            aaa = np.dot(aaa, H)

        PP = np.linalg.inv(np.matrix(P))
        PP = np.array(PP)

        for i in range(len(a)):
            for j in range(len(a)):
                a[i][j]=round(a[i][j],3)
        #for i in range(n):
         #   for j in range(n):
          #      PP[i][j] = round(PP[i][j], 3)

        """
        aaaaaa = np.dot(P, ccc)
        aaaaaa = np.dot(aaaaaa, np.transpose(P))

        PP = np.linalg.inv(np.matrix(P))
        PP = np.array(PP)

        dddd = np.dot(PP, a)
        dddd = np.dot(dddd, np.transpose(PP))
        dddd = np.array(dddd)

        for i in range(len(dddd)):
            M1 = np.array([dddd[i], ddd[i]])
            similarity1 = cosine_similarity(M1)
            x = 0

        ####
        vv, ww = np.linalg.eig(np.matrix(ddd))
        vv, ww = self.sort_eigens(vv, ww)
        vv2, ww2 = np.linalg.eig(np.matrix(a))
        vv2, ww2 = self.sort_eigens(vv2, ww2)

        ######
        ws = np.dot(PP, np.transpose(ww2))
        ws = np.transpose(ws)
        ws = ws.tolist()

        for i in range(len(dddd)):
            M1 = np.array([ws[i], ww[i]])
            similarity1 = cosine_similarity(M1)
            x = 0
        """

        return a, Pinv, P
    def perturbation_arrowhead(self, H):
        alpha = H[0][0]
        n = len(H)
        H2 = np.zeros(shape=(n, n))
        D = []
        z = []
        H2[0][0] = H[0][0]
        for i in range(n - 1):
            D.append(H[i + 1][i + 1])
            z.append(H[0][i + 1])
        D2 = sorted(D, reverse=True)
        z2 = []
        P = np.zeros(shape=(n, n))
        for i in range(n):
            P[i][i] = 1
        for i in range(len(D)):
            x = [index for index in range(len(D)) if D[index] == D2[i]]
            x = x[0]
            z2.append(z[x])
            z[x] = z[i]
            D[x] = D[i]
            z[i] = -10000000
            D[i] = -10000000
            # D.remove(D[0])
            # z.remove(z[0])
            temp = np.zeros(shape=(n, n))
            for j in range(n):
                if j != i + 1 and j != x + 1:
                    temp[j][j] = 1
            temp[i + 1][x + 1] = 1
            temp[x + 1][i + 1] = 1
            P = np.dot(temp, P)

        P = np.matrix(P)
        # P2 = np.zeros(shape=(n, n))
        # for i in range(n-1):
        #    P2[i][i+1] = 1
        # P2[-1][0]=1
        # P3 = np.dot(P2,P)




        PT = np.transpose(P)

        H = np.matrix(H)

        test = np.dot(P, H)
        H2 = np.dot(test, PT)
        # v1, w1 = np.linalg.eig(H)
        # v1, w1 = self.sort_eigens(v1, w1)
        # v2, w2 = np.linalg.eig(H2)
        # v2, w2 = self.sort_eigens(v2, w2)


        # v2m = np.zeros(shape=(n,n))
        # for i in range(n):
        #   v2m[i][i]= v2[i]
        # x = np.transpose(np.matrix(w2))
        # H3 = np.dot(x,v2m)
        # H3 = np.dot(H3,w2)
        ##########
        P2 = np.zeros(shape=(n, n))
        for i in range(n):
            P2[i][i] = 1
        for i in range(n - 1):
            temp = np.zeros(shape=(n, n))
            for j in range(n):
                if j != i and j != i + 1:
                    temp[j][j] = 1
            temp[i][i + 1] = 1
            temp[i + 1][i] = 1
            P2 = np.dot(temp, P2)

        HF = np.dot(P2, H2)
        HF = np.dot(HF, np.transpose(P2))
        HF = np.matrix(HF)

        # P2 = np.zeros(shape=(n,n))
        w11 = np.dot(PT, x)
        ##

        P3 = np.dot(P2, P)
        # v33, w33 = np.linalg.eig(HF)
        # v33, w33 = self.sort_eigens(v33,w33)
        # x = np.transpose(np.matrix(w33))
        # w12 = np.dot(np.transpose(P3),x)







        # for i in range(n - 1):
        #   H2[i][i] = D2[i]
        #  H2[i][-1] = z2[i]
        # H2[-1][i] = z2[i]
        # return H2

        return P3, alpha, D2, z2

    def bisect_arrowhead(self, D, z, alpha, side):
        n = len(D) + 1
        eps = 0.0001
        if side == -1:
            x = self.vec_length(z)
            left = min([item - x for item in D])
            left = min(left, alpha - self.vec_norm(z))
            right = min(D)
        else:
            left = max(D)
            x = self.vec_length(z)
            right = max([item + x for item in D])
            right = max(right, alpha + self.vec_norm(z))

        middle = (left + right) / 2
        while (right - left) / abs(middle) > 2.0 * eps:
            Fmiddle = 0
            for l in range(n - 1):
                Fmiddle += (z[l] * z[l]) / (D[l] - middle)
            Fmiddle = alpha - middle - Fmiddle
            if Fmiddle > 0:
                left = middle
            else:
                right = middle
            middle = (left + right) / 2

        return right

    def shiftAndInvert_arrowhead(self, D, z, alpha, k):
        n = len(D) + 1
        D2 = [item - D[k] for item in D]
        a = alpha - D[k]
        w1 = []
        w2 = []
        if z[k] == 0:
            z[k] = 0.001
        for l in range(k):
            if D2[l] == 0:
                D2[l] = 0.001
            w1.append(-(z[l] / D2[l]) / z[k])
        for l in range(k + 1, n - 1):
            if D2[l] == 0:
                D2[l] = 0.001
            w2.append(-(z[l] / D2[l]) / z[k])
        wc = 1 / z[k]
        invD1 = []
        invD2 = []
        for l in range(k):
            if D2[l] == 0:
                D2[l] = 0.001
            invD1.append(1 / D2[l])
        for l in range(k + 1, n - 1):
            if D2[l] == 0:
                D2[l] = 0.001
            invD2.append(1 / D2[l])
        sum1 = 0
        sum2 = 0
        for l in range(k):
            if D2[l] == 0:
                D2[l] = 0.001
            sum1 += (z[l] * z[l]) / D2[l]
        for l in range(k + 1, n - 1):
            if D2[l] == 0:
                D2[l] = 0.001
            sum2 += (z[l] * z[l]) / D2[l]
        b = (-a + sum1 + sum2) / (z[k] * z[k])

        return invD1, invD2, w1, w2, wc, b

        ###########
        """
        H1 = np.zeros(shape=(n, n))
        H2 = np.zeros(shape=(n, n))
        H3 = np.zeros(shape=(n, n))
        for i in range(n-1):
            H1[i][i] = D[i]
            H2[i][i] = D[i] - D[k]
        for i in range(len(invD1)):
            H3[i][i] = invD1[i]
        H3[k][k] = b
        for i in range(len(invD2)):
            H3[i+k+1][i+k+1] = invD2[i]
        H1[n-1][n-1] = alpha
        H2[n-1][n-1] = a
        for i in range(n-1):
            H1[i][- 1] = z[i]
            H1[-1][i] = z[i]
            H2[i][- 1] = z[i]
            H2[-1][i] = z[i]
        for i in range(len(invD1)):
            H3[i][k] = w1[i]
            H3[k][i] = w1[i]
        H3[k][-1] = wc
        H3[-1][k] = wc
        for i in range(len(invD2)):
            H3[i + k+1][k] = w2[i]
            H3[k][i + k+1] = w2[i]
        """
        # x1 = np.array(H2)
        # H5 = np.matrix(H2)
        # H4= np.linalg.inv(np.matrix(H2))
        # out = np.dot(H4,H5)
        # out = np.array(out)
        # for i in range(len(out)):
        #   for j in range(len(out)):
        #      out[i][j] = round(out[i][j],3)
        # x2 = np.array(H3)
        # out = np.dot(x1,x2)
        # out2= np.zeros(shape=(n, n))
        # for i in range(len(H2)):
        #   for j in range(len(H2)):
        #      out2[i][j] = round(out[i][j],3)

        # x=self.shiftAndInvert_arrowhead_old(H1,k)
        ##########

    def aheig_basic_arrowhead(self, k, D, alpha, z):
        n = len(D) + 1
        i = -1
        if k == 0:
            lamda = self.bisect_arrowhead(D, z, alpha, 1)
            lamdan = self.bisect_arrowhead(D, z, alpha, -1)
            if abs(lamdan / lamda) < 10:
                v = lamda
                w = self.vect_arrowhead(D, z, lamda)
            else:
                sigma = D[0]
                i = 0
                side = 1
        elif (k == n - 1):
            lamda = self.bisect_arrowhead(D, z, alpha, -1)
            lamda1 = self.bisect_arrowhead(D, z, alpha, 1)
            if abs(lamda1 / lamda) < 10:
                v = lamda
                w = self.vect_arrowhead(D, z, lamda)
            else:
                sigma = D[-1]
                i = n - 2
                side = -1
        else:
            Dtemp = [item - D[k] for item in D]
            atemp = alpha - D[k]
            middle = float(Dtemp[k - 1]) / 2.0
            Fmiddle = atemp - middle
            for l in range(n - 1):
                Fmiddle += (z[l] * z[l]) / (D[l] - middle)
            if Fmiddle < 0:
                sigma = D[k]
                i = k
                side = 1
            else:
                sigma = D[k - 1]
                i = k - 1
                side = -1
        if (i != -1):
            D1inv, D2inv, w1, w2, wc, b = self.shiftAndInvert_arrowhead(D, z, alpha, i)
            D2 = []
            z2 = []
            for j in range(len(D1inv)):
                D2.append(D1inv[j])
                z2.append(w1[j])
            D2.append(0)
            z2.append(wc)
            for j in range(len(D2inv)):
                D2.append(D2inv[j])
                z2.append(w2[j])
            v = self.bisect_arrowhead(D2, z2, b, side)
            if v == 0:
                v += 0.1
            m = 1.0 / float(v)
            xx = D - sigma
            w = self.vect_arrowhead([item - sigma for item in D], z, m)
            v = m + sigma
        return v, w

    def arrowheadmatrix(self, H):
        P, alpha, D, z = self.perturbation_arrowhead(H)
        PT = np.transpose(P)
        n = len(D) + 1
        vs = []
        ws = []
        for i in range(0, n):
            v, w = self.aheig_basic_arrowhead(i, D, alpha, z)
            vs.append(v)
            ws.append(w)
        ws = np.dot(PT, np.transpose(ws))
        ws = np.transpose(ws)
        ws = ws.tolist()
        ####
        # matrix1 = np.matrix(H)
        # D1, Q1 = np.linalg.eig(matrix1)
        # vs2, ws2 = self.sort_eigens(D1, Q1)
        ###
        # for i in range(len(vs2)):
        #    M1 = np.array([ws[i], ws2[i]])
        #    similarity1 = cosine_similarity(M1)
        #    if abs(similarity1[0][1])<0.99:
        #        y=0
        #    x = 0

        # 3z = 0

        return vs, ws

    def divideConqure(self, T, k, bool1, ep):
        n = len(T)
        if n > k:
            n1 = n / 2
            n2 = n - n1 - 1
            T1 = np.zeros(shape=(n1, n1))
            T2 = np.zeros(shape=(n2, n2))
            for i in range(n1):
                for j in range(n1):
                    T1[i][j] = T[i][j]
            for i in range(n2):
                for j in range(n2):
                    T2[i][j] = T[i + n1 + 1][j + n1 + 1]
            D1, Q1 = self.divideConqure(T1, k, bool1, ep)
            Q1 = np.transpose(Q1)
            D2, Q2 = self.divideConqure(T2, k, bool1, ep)
            Q2 = np.transpose(Q2)
            Q2 = np.array(Q2)
            A1 = np.zeros(shape=(n, n))
            A2 = np.zeros(shape=(n, n))
            H = np.zeros(shape=(n, n))
            for i in range(n1):
                for j in range(n1):
                    # print("*****")
                    # print("i = " + str(i))
                    # print("j = " + str(j))
                    # print("len Q1 = " + str(len(Q1)))
                    # print("len Q1[i] = " + str(len(Q1[i])))
                    if len(Q1[i]) < n1:
                        v, w = np.linalg.eig(np.matrix(T1))
                        v, w = self.sort_eigens(v, w)
                        x = 0

                    A1[i][j + 1] = Q1[i][j]
                    A2[j + 1][i] = Q1[i][j]
            for i in range(n2):
                for j in range(n2):
                    if len(Q2[i]) < n2:
                        v, w = np.linalg.eig(T2)
                        v, w = self.sort_eigens(v, w)
                        x = 0

                    # print("*****")
                    # print("i = " + str(i))
                    # print("j = " + str(j))
                    # print("n1 = " + str(n1))
                    # print("n2 = " + str(n2))
                    # print("len Q2 = " + str(len(Q2)))
                    # print("len Q2[i] = " + str(len(Q2[i])))
                    A1[i + n1 + 1][j + n1 + 1] = Q2[i][j]
                    A2[j + n1 + 1][i + n1 + 1] = Q2[i][j]
            A1[n1][0] = 1
            A2[0][n1] = 1
            H[0][0] = T[n1][n1]
            for i in range(n1):
                H[i + 1][i + 1] = D1[i]
            for i in range(n2):
                H[i + n1 + 1][i + n1 + 1] = D2[i]

            for i in range(n1):
                H[0][1 + i] = Q1[-1][i] * T[n1 - 1][n1]
                H[1 + i][0] = H[0][1 + i]
            for i in range(n2):
                H[0][1 + n1 + i] = Q2[0][i] * T[n1 + 1][n1]
                H[1 + n1 + i][0] = H[0][1 + n1 + i]

            #######change
            vH, wH = self.arrowheadmatrix(H)
            U = wH
            UT = np.transpose(U)
            left = np.dot(A1, np.array(UT))
            QH = np.transpose(left)
        else:
            if bool1:
                matrix1 = np.matrix(T)
                obj = EigenPair()
                obj.update_parameters(matrix1,isMatrix=True)
                obj.eigen_pairs(ep)
                D2 = obj.vals
                Q2 = obj.vecs
            else:
                matrix2 = np.matrix(T)
                D2, Q2 = np.linalg.eig(matrix2)
                D2, Q2 = self.sort_eigens(D2, Q2)
                for i in range(len(D2)):
                    D2[i] = D2[i].real
                    for j in range(len(D2)):
                        Q2[i][j] = Q2[i][j].real
            vH = D2
            QH = Q2
            if len(vH) < len(T) or len(QH[0]) < len(T) or len(QH) < len(T):
                obj = EigenPair()
                matrix2 = np.matrix(T)
                obj.update_parameters(matrix2,isMatrix=True)
                obj.eigen_pairs(ep)
                vH = obj.realVals
                QH = obj.realVecs

            Q = np.transpose(QH)
            if len(Q[0]) < n:
                x = 0

        return vH, QH

    def eigen_pairs_divideConqure(self, ep, size, bool1):
        start = time.clock()
        obj = EigenPair()
        T2 = self.A
        ###
        v1, w1 = np.linalg.eig(np.matrix(T2))
        v1, w1 = obj.sort_eigens(v1, w1)
        a22, Pt, P = obj.householder_transformation4(T2)
        v3, w3 = np.linalg.eig(np.matrix(a22))
        v3, w3 = obj.sort_eigens(v3, w3)
        ws = np.dot(Pt, np.transpose(w3))
        ws = ws.transpose()
        for i in range(len(v1)):
            M1 = np.array([w1[i], ws[i]])
            similarity1 = cosine_similarity(M1)
            x = 0
        ####

        T, PH ,P= obj.householder_transformation4(T2)
        v1, w1 = obj.divideConqure(T, size, bool1, ep)
        w1 = np.dot(PH, np.transpose(w1))
        w1 = np.transpose(w1)
        w1 = w1.tolist()
        self.vals = v1
        self.vecs = w1
        self.time = time.clock() - start

    def eigen_pairs_QR(self):
        #start = time.clock()
        #T2 = self.A
        v1, w1 = self.QR_transformation()
        self.QRVals = v1
        self.QRVecs = w1
        #self.time = time.clock() - start

    def log4(self, datadir, graphname, ep):
        self.evaluatePercents(ep)
        with open(datadir + "output/" + graphname + "(" + str(ep) + ")" + "dCStandard.csv", 'wb') as csvfile:
            fieldnames = ["name"]
            for i in range(self.dimension):
                fieldnames.append(str(i))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            dict0 = {"name": "realval"}
            dict1 = {"name": "realvec"}
            dict2 = {"name": "appval"}
            dict3 = {"name": "appvec"}
            dict4 = {"name": "percents"}
            dict5 = {"name": "valErros"}
            dict6 = {"name": "vecErrors"}
            dict7 = {"name": "totalErrors"}
            dict8 = {"name": "time"}

            for i in range(len(self.realVals)):
                dict0.update({str(i): self.realVals[i]})
                dict1.update({str(i): self.realVecs[i]})
            for i in range(len(self.vals)):
                dict2.update({str(i): self.vals[i]})
                dict3.update({str(i): self.vecs[i]})
            dict4.update(self.percents)
            dict5.update(self.valErrors)
            dict6.update(self.vecErrors)
            dict7.update(self.totalErrors)
            dict8.update({'0': self.time})
            writer.writerow(dict0)
            writer.writerow(dict1)
            writer.writerow(dict2)
            writer.writerow(dict3)
            writer.writerow(
                {"name": "percents", '0': "5", '1': "10", '2': "15", '3': "20", '4': "25", '5': "30", '6': "35",
                 '7': "40", '8': "45", '9': "50", '10': "55", '11': "60", '12': "65", '13': "70", '14': "75",
                 '15': "80", '16': "85", '17': "90", '18': "95", '19': "100"})
            writer.writerow(dict4)
            writer.writerow(dict5)
            writer.writerow(dict6)
            writer.writerow(dict7)
            writer.writerow(dict8)

    def log5(self, datadir, graphname, ep):
        self.evaluatePercents(ep)
        with open(datadir + "output/" + graphname + "(" + str(ep) + ")" + "DC.csv", 'wb') as csvfile:
            fieldnames = ["name"]
            for i in range(self.dimension):
                fieldnames.append(str(i))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            dict0 = {"name": "realval"}
            dict1 = {"name": "realvec"}
            dict2 = {"name": "appval"}
            dict3 = {"name": "appvec"}
            dict4 = {"name": "percents"}
            dict5 = {"name": "valErros"}
            dict6 = {"name": "vecErrors"}
            dict7 = {"name": "totalErrors"}
            dict8 = {"name": "time"}

            for i in range(len(self.realVals)):
                dict0.update({str(i): self.realVals[i]})
                dict1.update({str(i): self.realVecs[i]})
            for i in range(len(self.vals)):
                dict2.update({str(i): self.vals[i]})
                dict3.update({str(i): self.vecs[i]})
            dict4.update(self.percents)
            dict5.update(self.valErrors)
            dict6.update(self.vecErrors)
            dict7.update(self.totalErrors)
            dict8.update({'0': self.time})
            writer.writerow(dict0)
            writer.writerow(dict1)
            writer.writerow(dict2)
            writer.writerow(dict3)
            writer.writerow(
                {"name": "percents", '0': "5", '1': "10", '2': "15", '3': "20", '4': "25", '5': "30", '6': "35",
                 '7': "40", '8': "45", '9': "50", '10': "55", '11': "60", '12': "65", '13': "70", '14': "75",
                 '15': "80", '16': "85", '17': "90", '18': "95", '19': "100"})
            writer.writerow(dict4)
            writer.writerow(dict5)
            writer.writerow(dict6)
            writer.writerow(dict7)
            writer.writerow(dict8)

    def evaluateQR(self, ep):
        correctList = []
        for i in range (len(self.QRVals)):
            vals1 = self.QRVals[i]
            vecs1 = self.QRVecs[i]
            for item in self.realVals:
                correctList.append(item)
            correct = 0
            valDist = 0
            vecDist = 0

            for i in range(len(self.vals)):
                appVal = vals1[i]
                appVec = vecs1[i]

                correctList, value = self.find_closest_eigen(correctList, appVal, ep)
                if value != -10000 and value != 10000:
                    correct += 1
                    vvv = abs((value - appVal) / appVal)
                    if (vvv > 2 and abs(appVal) > 0 and abs(appVal) < 1 ):
                        valDist += 0.5
                    else:
                        valDist += abs((value - appVal) / appVal)

                    dist = self.matVecMul(self.A - self.I * value, appVec)
                    vecDist += self.vec_dist(dist, [0] * self.dimension)

            self.percentsQR.append( float(correct) / float(self.dimension))
            self.vecErrorsQR.append(vecDist / self.dimension)
            self.valErrorsQR.append(valDist / self.dimension)
            self.totalErrorsQR.append(((vecDist / self.dimension) + (valDist / self.dimension)) / 2 )
    def log6(self, datadir, graphname, ep):
        self.evaluate(1)
        self.evaluateQR(1)

        with open(datadir + "output/" + graphname + "(" + str(ep) + ")" + "DC-QR.csv", 'wb') as csvfile:
            fieldnames = ["name"]
            for i in range(self.dimension):
                fieldnames.append(str(i))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            dict0 = {"name": "percents"}
            dict1 = {"name": "valErrosDC"}
            dict2 = {"name": "vecErrorsDC"}
            dict3 = {"name": "totalErrorsDC"}
            dict4 = {"name": "timeDC"}

            dict9 = {"name": "percentsQR"}
            dict5 = {"name": "valErrosQR"}
            dict6 = {"name": "vecErrorsQR"}
            dict7 = {"name": "totalErrorsQR"}
            dict8 = {"name": "timeQR"}

            dict0.update({'0': self.percent})
            dict1.update({'0':self.valError})
            dict2.update({'0':self.vecError})
            dict3.update({'0':self.totalError})
            dict4.update({'0': self.time})

            for i in range(len(self.timesQR)):
                dict9.update({str(i): self.percentsQR[i]})
                dict5.update({str(i): self.valErrorsQR[i]})
                dict6.update({str(i): self.vecErrorsQR[i]})
                dict7.update({str(i): self.totalErrorsQR[i]})
                dict8.update({str(i): self.timesQR[i]})


            writer.writerow(dict0)
            writer.writerow(dict1)
            writer.writerow(dict2)
            writer.writerow(dict3)
            writer.writerow(dict4)
            writer.writerow(dict9)
            writer.writerow(dict5)
            writer.writerow(dict6)
            writer.writerow(dict7)
            writer.writerow(dict8)

    def householder_reflection(self, A):
        """Perform QR decomposition of matrix A using Householder reflection."""
        (num_rows, num_cols) = np.shape(A)

        # Initialize orthogonal matrix Q and upper triangular matrix R.
        Q = np.identity(num_rows)
        R = np.copy(A)

        # Iterative over column sub-vector and
        # compute Householder matrix to zero-out lower triangular matrix entries.
        for cnt in range(num_rows - 1):
            x = R[cnt:, cnt]

            e = np.zeros_like(x)
            e[0] = copysign(np.linalg.norm(x), -A[cnt, cnt])
            u = x + e
            v = u / np.linalg.norm(u)

            Q_cnt = np.identity(num_rows)
            Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, v)

            R = np.dot(Q_cnt, R)
            Q = np.dot(Q, Q_cnt.T)

        return (Q, R)

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


