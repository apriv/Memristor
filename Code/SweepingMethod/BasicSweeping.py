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


def eigen_pairs(matrix, k, ep, max1, min, v):
    #step1
    v1,ys,w1 = eigen_values_computation(matrix, k, ep, max1, min)
    #step2
    v2 = eigenvalue_computation_duplicates(matrix, k, ep, min, max1, v1)
    #step3
    v3, w3 = eigenvector_computation(matrix, v2, len(v))
    return v3, w3

def eigen_values_computation(input, k, ep, max1, min):
    print("min = " +str(min))
    print("max = " + str(max1))
    print("len = " + str(len(input)))
    eigvec=[]
    from numpy import linalg as LA
    A = np.mat(input)
    b = first_n_Primes(len(input))
    b2 = shuffled_n_Primes(len(input))
    I = np.identity(len(input))
    x_vals = []
    y_vals = []
    xs =[]
    ys =[]
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
        eig = find_max(result)
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

            result1 = LA.solve(A - I*(max1+ep+ep), b2)
            r1=find_max(result1)
            result2 = LA.solve(A - I*(max1+ep), b2)
            r2=find_max(result2)
            result3 = LA.solve(A - I*(max1), b2)
            r3=find_max(result3)
            if (r2>r1) and (r2>r3):
                eigenvalues.append(round(max1.real + ep, 15))
                ys.append( values[values.index(eig) - 1])
                xs.append(round(max1.real + ep, 15))
                eigvec.append(result)
                j += 1

        if (len(eigenvalues) >= k or max1 < min - 1):
            bool = False
        i += 1

        max1 -= ep
        if max1 < min - 1:
            break
    return xs,ys,eigvec

def eigen_values_computation2(input, k, ep, max1, min):
    print("min = " +str(min))
    print("max = " + str(max1))
    print("len = " + str(len(input)))
    eigvec=[]
    from numpy import linalg as LA
    A = np.mat(input)
    b = first_n_Primes(len(input))
    b2 = shuffled_n_Primes(len(input))
    I = np.identity(len(input))
    x_vals = []
    y_vals = []
    xs =[]
    ys =[]
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
        eig = find_max(result)
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
            eigenvalues.append(round(max1.real + ep, 15))
            ys.append(values[values.index(eig) - 1])
            xs.append(round(max1.real + ep, 15))
            eigvec.append(result)
            j += 1
        if (len(eigenvalues) >= k or max1 < min - 1):
            bool = False
        i += 1
        max1 -= ep
        if max1 < min - 1:
            break
    return xs,ys,eigvec



def eigenvalue_computation_duplicates(input,k,ep,min,max1, eigens):
    A = np.mat(input)
    xs1= eigens
    output = []
    x1 = []
    xx1 = []
    for item in xs1:
        output.append(item)
        x1.append(item)
        xx1.append(item)
    ep2 = 10 * ep
    counter = 0
    counter2= 0
    while len(output) < k:
        counter2+=1
        b = False
        B = np.random.uniform(0, ep2, len(A) * len(A)).reshape(len(A), len(A))
        A = A + B
        matrix = np.matrix(A)
        xs2,w2,ys = eigen_values_computation2(matrix, k, ep, max1+1, min)
        x2 = []
        if len(xs2) > len(output):
            counter = 1
            for item in xs2:
                x2.append(item)
            for item in xs1:
                xs2, y = find_closest_eigen(xs2, item, 2)
            for item in xs2:
                xx1, y = find_closest_eigen(xx1, item, 2)
                xx1.append(y)
                output.append(y)
                if len(output) == len(A):
                    break
            if counter == 1:
                break
            if counter2 >= 10:
                break
            xs1 = []
            for item in output:
                xs1.append(item)
    output = sorted(output, reverse=True)
    return output


def eigenvector_computation(input,xs,ll):
    vectors =[]
    values =[]
    eigenvalues = xs
    A = np.mat(input)
    I = np.identity(len(input))
    i = 0
    k = 0
    lens=len(eigenvalues)
    while(True):
        if (eigenvalues[i] not in values):
            print("counter = " + str(k))
            item = eigenvalues[i].real + 0.0001
            if i < len(eigenvalues) - 1 and (eigenvalues[i].real == eigenvalues[i + 1].real):
                a,b,c =eigen_values_computation(input, 1, 0.0001, eigenvalues[i].real + 0.1 , min)

                # vector = null2(A - item *I )
                vector = null(A - a[0].real * I)
                vector = np.array(vector)
                vector2 = vector.flatten()
                x = vector2.tolist()
                y = len(x)
                while (y == 0):
                    item = item + 0.0001
                    vector = null(A - (item) * I)
                    vector = np.array(vector)
                    vector2 = vector.flatten()
                    x = vector2.tolist()
                    y = len(x)
                x = vector[:, 0]
                y = vector[0, :]
                l = len(vector[0, :])
                for j in range(l):

                 #   new_matrix[:, k] = vector[:, j]
                    vectors.append(vector[:, j].tolist())
                    values.append(eigenvalues[i].real)
                    k+=1
                    lens += (l-1)

            else:
                vector, x = inverse_power(A - item * I)

                #new_matrix[:, k] = vector
                vectors.append(vector)
                values.append(eigenvalues[i].real)
                k+=1
        i+=1
        if  i>=len(eigenvalues) or k>=ll:
            break

    #new_matrix2 = np.ndarray(shape=(len(input), len(values)), dtype=float, order='F')
    #for i in range(len(values)):
        #new_matrix2[:, i] = new_matrix[:, i]

    return  values, vectors

def first_n_Primes(n):
    number_under_test = 4
    primes = [2,3]
    while len(primes) < n:
        check = False
        for prime in primes:
            if prime > math.sqrt(number_under_test) : break
            if number_under_test % prime == 0:
                check = True
                break
        if not check:
            for counter in range(primes[len(primes)-1],number_under_test-1,2):
                if number_under_test % counter == 0:
                    check = True
                    break
        if not check:
            primes.append(number_under_test)
        number_under_test+=1
    return primes

def find_closest_eigen(eig_val,eig,ep):
    output=-10000
    out = eig_val
    min = 10000
    counter = 0
    for item in eig_val:
        counter += 1
        dist = abs(item - eig)
        if dist < min:
            output = item
            min = dist
    if min < ep:
        out.remove(output)
    else:
        output = -10000
    return out,output

def sort_eigens(v1, w1):
    w2 = []
    v2 = sorted(v1, reverse=True)
    v3 =[]
    for i in range (0,len(v1)):
        v3.append(v2[i])
        x= v2[i]
        b = [item for item in range(len(v1)) if v1[item] == x]
        l = w1[:, b]
        l = l[:, 0]
        l = l.flatten()
        l = l.tolist()
        l = l[0]
        w2.append(l)
    #w2 = np.matrix(w2)
    #w3 = w2.transpose()
    return v3, w2

def find_max(input):
    max1=0
    for element in input:
        if abs(element)>max1:
            max1=abs(element)
    return max1

def shuffled_n_Primes(n):
    vec2 =[]
    vec1 = first_n_Primes(n)
    while len(vec1)>1:
        x =  random.randint(0,len(vec1)-1)
        vec2.append(vec1[x])
        vec1.remove(vec1[x])
    vec2.append(vec1[0])
    return vec2

def null(A, eps=1e-3):
  u,s,vh = svd(A,full_matrices=1,compute_uv=1)
  null_space = compress(s <= eps, vh, axis=0)
  vector = np.array(null_space)
  vector2 = vector.flatten()
  x = vector2.tolist()
  y = len(x)
  vh = np.array(vh)
  null_spacetest = vh[len(vh) - 1, :]
  if y ==0:
      vh = np.array(vh)
      null_space = vh[len(vh)-1,:]
      null_space = np.matrix(null_space)
  return null_space.T

def inverse_power(E):
    MATRIX_SIZE = len(E)

    rx = np.random.rand(1, MATRIX_SIZE)
    r0 = rx[0]
    w0 = np.linalg.solve(E, r0)
    #a = abs(w0)
    #a = a.tolist()
    #b = max(a)
    r = w0 / max(abs(w0))

    w = np.linalg.solve(E, r)

    # Start the inverse_power until convergence
    M = np.array([w0, w])
    # print similarities
    count = 0
    #while (similarities[0][1] < 0.9999999995):
    while (count<10):
        w0 = w
        w = np.linalg.solve(E, r)
        r = w / max(abs(w))
        count = count + 1
    sum=0
    for item in w:
        sum+=abs(item)*abs(item)
    sum= sqrt(sum)
    w2=[]
    neg=1
    if w[0] <0:
        neg= -1
    for item in w:
        w2.append(round(item/sum,8)*neg)
    return w2, count


def sort_eigens2(v1, w1):
    w2 = []
    v2 = sorted(v1, reverse=True)
    v3 =[]
    for i in range (0,len(v1)):
        if v2[i] not in v3:
            v3.append(v2[i])
            x = v2[i]
            b = [item for item in range(len(v1)) if v1[item] == x]
            l = w1[:, b]
            l = l.flatten()
            l = l.tolist()
            l = l[0]
            w2.append(l)
    #w2 = np.matrix(w2)
    #w3 = w2.transpose()
    return v3, w2


def per_correct_eigens (real, app, ep, realvec, appvec):
    vecsim =0
    real2= []
    errorvec =0
    error =0
    for item in real:
        real2.append(item)
    n=0
    l = len(real)
    for item in app:
        real2, val = find_closest_eigen(real2, item, ep )
        if val != -10000:
            n+=1
            indice1 = [i for i in range(len(real)) if real[i] == val]
            indice2 = [i for i in range(len(app)) if app[i] == item]
            indice1 = indice1[0]
            indice2 = indice2[0]
            a1 = item
            a2 = val.real
            a2 = round(a2,10)
            if a2 ==0:
                e = abs(a1)
            else:
                e = a1 - a2
                e = e / a2
                e = abs(e)

            error += e

            vec1 = realvec[indice1]
            vec1 = np.array(vec1)
            vec1 = vec1.flatten()
            vec1 = vec1.tolist()
            #vec1 = vec1[0]
            vec2 = appvec[indice2]
            vec2 = vec2.tolist()
            M = np.array([vec1,vec2])
            sim = cosine_similarity(M)
            vecsim+=sim[0][1]
            error2 = 0
            for i in range(len(vec1)):
                a1 = vec1[i].real
                a1 = round(a1, 10)
                if a1 == 0:
                    e = abs(a1)
                else:
                    a2 = vec2[i]
                    e = abs(a1) - abs(a2)
                    if a1 == 0:
                        a1 = 0.0000000001
                    e = e / a1
                    e = abs(e)

                error2 += e
            error2 = error2 / len(vec1)
            errorvec += error2
    error = error / n
    errorvec = errorvec / n
    vecsim= vecsim/n

    output = double(n)/l

    return output,error,errorvec,vecsim



def  eigen_pairs_perturbation(datadir, graphNames):

    eps =[ 0.0001]
    for graphname in graphNames:
        G = nx.read_edgelist(datadir + "dataset/" + graphname + ".txt")
        print ("*********************dataset =" + graphname )
        print ("read....")
        matrix = nx.adjacency_matrix(G)
        A = matrix.toarray()
        print ("eigenval....")

        matrix = np.matrix(A)

        v1, w1 = np.linalg.eig(matrix)
        v2, w2 = sort_eigens(v1, w1)
        v3, w3 = sort_eigens2(v1, w1)

        max1 = v1[0] + 1
        min = v2[len(v2) - 1]
        for ep in eps:
            with open(datadir + "output/" + graphname + "(" + str(ep) + ")" + "result.csv", 'wb') as csvfile:
                fieldnames = ["name"]
                eigenvals = []
                eigenvecs = []
                # step 1 (sweeping method)
                start = time.clock()
                k = len(v2)
                v,w = eigen_values_computation(matrix, k, ep, max1, min,v3, w3)
                first = time.clock() - start
                # step 3 (finding duplicates)
                xs3 = eigenvalue_computation_duplicates(matrix, k, ep, min, max1, v,v2)
                second = time.clock() - first
                # step 4 (inverse power)
                xs3, eigenvectors = eigenvector_computation(matrix, xs3 ,w ,v ,len(v2),v2,w2)
                third = time.clock() - second
                eigenvals.append(xs3)
                eigenvecs.append(eigenvectors)
                percent = [5, 10, 15, 20, 25, 30, 50, 100]
                dict6 = {"name": "percent"}
                dict7 = {"name": "error"}
                dict8 = {"name": "errorvecs"}
                dict9 = {"name": "totalerror"}

                cc=0
                for item in percent:
                    newv2 =[]
                    newxs3 =[]
                    neww2 =[]
                    neweigenvectors =[]
                    s1= float(len(v2))
                    s2 = float(item)
                    x= (float(len(v2))*float(item))/100
                    x = int (x)
                    for k in range(x):
                        newv2.append(v2[k])
                        neww2.append(w2[k])
                        if k < len(xs3):
                            newxs3.append(xs3[k])
                            neweigenvectors.append(eigenvectors[:, k])

                    per1 ,error, error2, vecsim= per_correct_eigens(newv2, newxs3, ep,neww2, neweigenvectors)

                    dict6.update(({str(cc): per1}))
                    dict7.update(({str(cc): error}))
                    dict8.update(({str(cc): error2}))
                    dict9.update(({str(cc): (error2+error)/2}))


                    cc+=1

                maxlen = len(v1)
                if maxlen <len(v):
                    maxlen = len(v)
                if maxlen <len(xs3):
                    maxlen = len(xs3)

                for i in range(maxlen):
                    fieldnames.append(str(i))
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                dict1 = {"name": "realval"}
                dict0 = {"name": "realvec"}
                dict2 = {"name": "step1"}
                dict4 = {"name": "step3"}
                dict5 = {"name": "step4"}

                for item in range(len(v2)):
                    dict1.update({str(item): v2[item]})
                    dict0.update({str(item): w2[item]})
                for item in range(len(v)):
                    dict2.update({str(item): v[item]})
                for item in range(len(xs3)):
                    dict4.update({str(item): xs3[item]})
                    dict5.update({str(item): eigenvectors[item]})
                writer.writerow(dict1)
                writer.writerow(dict0)
                writer.writerow(dict2)
                writer.writerow(dict4)
                writer.writerow(dict5)
                writer.writerow({"name": "time(sec)", "0": first, "1": second, "2": third})
                writer.writerow({"name": "time(min)", "0": first/60, "1": second/60, "2": third/60, })
                writer.writerow({"name": "title", "0": 5, "1": 10, "2": 15, "3": 20, "4": 25, "5": 30, "6": 50, "7": 100})
                writer.writerow(dict6)
                writer.writerow(dict7)
                writer.writerow(dict8)
                writer.writerow(dict9)
               # writer.close()
                M = np.array([v2, xs3])
                if len(v2)==len(xs3):
                    similarities = cosine_similarity(M)
                    if abs(similarities[0][1])>0.9:
                        break


    return eigenvals, eigenvecs


if __name__ == '__main__':
    datadir = "/../"
    graphNames = ["karate", "dolphins", "lesmis", "squeak", "CA-GrQc", "robots", "fb", "lesmis", "celegansneural",
                  "power", "adjnoun", "polblogs", "polbooks", "netscience", "as-22july06", "G100", "G200", "G500",
                  "G1000", "G2000"]
    graphNames = ["squeak"]
    print ("start")
    for graphname in graphNames:
        G = nx.read_edgelist(datadir + "dataset/" + graphname + ".txt")
        print ("*********************dataset =" + graphname)
        print ("read....")
        matrix = nx.adjacency_matrix(G)
        A = matrix.toarray()
        print ("eigenval....")
        matrix = np.matrix(A)
        v1, w1 = np.linalg.eig(matrix)
        v1 = [item.real for item in v1]
        v2, w2 = sort_eigens(v1, w1)
        ep= 0.001
        max1 = v2[0]
        min = v2[len(v2)-1]
        v, w = eigen_pairs(matrix, len(v1) , ep, max1, min, v2)
