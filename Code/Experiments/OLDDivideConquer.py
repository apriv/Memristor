import networkx as nx
import metis
import numpy as num
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
import csv
import codecs
import time
import math
import numpy as np



def block_lanczos(A,V0,ep):
    #B0 = np.ndarray(shape=(len(A),len(A)), dtype=float, order='F')
    B0 = create_empty_ndarray(len(V0[0]),len(V0[0]))
    Q0 = create_empty_ndarray(len(A),len(V0[0]))
    Q1 = V0
    #eig_val_cov, eig_vec_cov = np.linalg.eig(A)
    #Q1 = eig_vec_cov
    QH =Q1
    l= len(A)



    bool = True
    counter =0
    while (bool):
        R = np.dot(A,Q1) - np.dot(Q0 , B0.transpose())
        A1 = np.dot( Q1.transpose(), R )
        R = R - np.dot(Q1,A1)
        Q2 ,B1 = np.linalg.qr(R)

        TH1 = np.dot(np.dot(QH.transpose() , A),QH)
        v1, w1 = np.linalg.eig(TH1)
        w33 = np.dot(QH,w1)
        v33, w33 = sort_eigens_new(v1, w33, len(QH[0]))

        #w1 = np.dot(QH,w.transpose())
        v2, w2 = sort_eigens2_new(v1, w1, len(QH[0]))
        v1, w1 = sort_eigens_new(v1, w1, len(QH[0]))
        w1 = np.dot(QH,w1)
        w2 = np.dot(QH,w2)
        w1= w1.transpose()
        w2=w2.transpose()
        d = check_distance(v1,w1,A)
        if counter>3:
            bool = False
        else:
            counter+=1
            Q0 = Q1
            Q1 = Q2
            QH = np.concatenate((Q2, QH), axis=1)
            B0 = B1
    return v2, w2

def check_distance(v1,w1,A):
    d = 0
    l = len(v1)
    if len(v1)>len(w1):
        l = len(w1)
    for i in range(0, l):
        x= np.dot(A , w1[i,:])
        y = np.dot(v1[i] , w1[i,:])
        d= d + np.dot(A , w1[i,:]) - np.dot(v1[i] , w1[i,:])
    d = d / l
    return vec_length(d)

def compare(graphNames, datadir,k):
    for graphname in graphNames:
        with open(datadir +"output/compare/"+graphname+ "newpartition.csv", 'wb') as csvfile:
            fieldnames = ["value","vector","1","2","3","4","5"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            G = nx.read_edgelist(datadir + "dataset/" + graphname + ".txt")
            G = pre_process(G)
            matrix = nx.adjacency_matrix(G)
            A = matrix.toarray()
            matrix = np.matrix(A)
            #k = len(G.nodes())/10
            start = time.time()
            list = []
            Graphs = []
            for p in range(0, k):
                list2 = []
                list.append(list2)
            (edgecuts, parts) = metis.part_graph(G, k)
            x = G.nodes()
            for i, p in enumerate(parts):
                x = G.nodes()
                list[p].append(G.nodes()[i])
            listv1 =[]
            listw1 =[]
            first = time.time()
            secondTime=0
            for p in range(0, k):
                H = G.subgraph(list[p])
                A1 = nx.adjacency_matrix(H).toarray()
                #v1, w1 = np.linalg.eig(A1)
                eig_val_cov, eig_vec_cov = np.linalg.eig(A1)
                eig_val_cov = sorted(eig_val_cov, reverse=True)
                max = eig_val_cov[0]
                min = eig_val_cov[len(eig_val_cov) - 1]
                ep = find_Distance(eig_val_cov)
                tempTime1 = time.time()
                #########################################
                v1, w1, eigenVectors = eigen_computation(A1, len(H.nodes())/5, ep, max, min)
                #v1, w1 = np.linalg.eig(A1)
                Graphs.append(w1)

                #listv1.append(v1)
                #listw1.append(w1)
                secondTime = time.time() - tempTime1 +secondTime

            second = time.time()
           # for p in range(0, k):
            #    v1, w1 = sort_eigens(listv1[p], listw1[p])
                # output.append(w1.transpose())
                #w1 = w1.flatten()
                #w1 = w1.tolist()
              #  Graphs.append(w1)
            third = time.time()
            V0 = direct_sum(Graphs)
            fourth = time.time()
            #v, V0 = np.linalg.eig(matrix)
            v1, w1 = block_lanczos(A, V0, 1)
            fifth = time.time()
            v2, w2 = np.linalg.eig(A)
            v2,w2 = sort_eigens2(v2,w2)

            similarity1,similarity2 = similarity(v1, v2, w1, w2)

            #similarity2 = similarity2 / len(v1)
            writer.writerow({"value": similarity1[0][1],"vector" :similarity2[0][1]})
            writer.writerow({"1": str(first-start),"2" :str(secondTime),"3": str(third-second),"4" :str(fourth-third),"5" :str(fifth-fourth)})

def create_empty_ndarray(d1,d2):
    output = np.ndarray(shape=(d1, d2), dtype=float, order='F')
    for i in range(d1):
        for j in range (d2):
            output[i][j]=0.0
    return output

def direct_sum(list):
    item1 = list[0]
    for i in range(1,len(list)):
        item2 = list[i]
        item1 = direct_sums(item1,item2)
    return item1

def direct_sums(a,b):
    dsum = num.zeros(num.add(a.shape, b.shape))
    dsum[:a.shape[0], :a.shape[1]] = a
    dsum[a.shape[0]:, a.shape[1]:] = b
    return dsum

def eigen_computation(input,k,ep,max,min):
    from numpy import linalg as LA
    A= np.mat(input)
    b = first_n_Primes(len(input))
    #max= max_lambda_values(input)+2
    I = np.identity(len(input))

    eig_val_cov, eig_vec_cov = np.linalg.eig(A)
    eig_val_cov = sorted(eig_val_cov,reverse=True)
    #ep=0.0001
    #ep = 0.5
    values=[]
    values.append(1000)
    values.append(1000)
    eigenvalues=[]
    bool=True
    while(bool):
        YI=I*max
        X= A-YI
        result = LA.solve(A-YI, b)
        eig=find_max(result)
        values.append(eig)
        if values[values.index(eig)] == values[values.index(eig) - 1]:
            values.remove(eig)
        elif values[values.index(eig)-1]> values[values.index(eig)-2] and values[values.index(eig)-1]> values[values.index(eig)]:
            eigenvalues.append(round(max+ep,5))
            if(len(eigenvalues)==k):
                bool=False
        max-=ep
        if max<min-1:
            eigenvalues.append(0)
            if (len(eigenvalues) == k):
                bool = False
       # if max<0:
        #    break
    eigenvectors=[]
    eigenvectors2=[]
    list = []
    list2 = []
    len1 =len(eigenvalues)
    len1 = len1 +0.0001
    new_matrix = np.ndarray(shape=(len(eig_val_cov),len(eigenvalues)), dtype=float, order='F')
    new_matrix2 = np.ndarray(shape=(len(eig_val_cov),len(eigenvalues)), dtype=float, order='F')

    counter = 0
    for item in eigenvalues:
        item += 0.1
        item2 = eig_val_cov[counter].real + 0.1
        vector, x = inverse_power(A - item * I)
        vector2, x2 = inverse_power(A - item2 * I)

        new_matrix[:, counter] = vector
        new_matrix2[:, counter] = vector2
        counter+=1



        #eigenvectors2.append(vector)
        #list.append(vector)
        #list2.append(vector2)


  #  new_matrix = np.matrix(new_matrix)
   # new_matrix = new_matrix.transpose()

#    new_matrix2 = np.matrix(new_matrix2)
 #   new_matrix2= new_matrix2.transpose()


    return eigenvalues, new_matrix, new_matrix2

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

def find_max(input):
    max=0
    for element in input:
      #  element = math.ceil(element * 100) / 100
        if abs(element)>max:
            max=abs(element)
    return max

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

def inverse_power(E):
    MATRIX_SIZE = len(E)

    rx = np.random.rand(1, MATRIX_SIZE)
    #r0= first_n_Primes(MATRIX_SIZE)
    r0 = rx[0]
    w0 = np.linalg.solve(E, r0)
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
        M = np.array([w0, w])
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

def paritition_graph(G,k):
    list=[]
    output=[]
    for p in range (0,k):
        list2=[]
        list.append(list2)
    (edgecuts, parts) = metis.part_graph(G, k)
    x = G.nodes()
    for i, p in enumerate(parts):
        x= G.nodes()
        list[p].append(G.nodes()[i])
    for p in range(0, k):
        H=G.subgraph(list[p])
        A=nx.adjacency_matrix(H).toarray()
        v1, w1 = np.linalg.eig(A)
        v1, w1 = sort_eigens(v1, w1)
        # output.append(w1.transpose())
        output.append(w1)
    return output

def pre_process(G):
    output = nx.Graph()
    nodes = []
    dict={}
    counter = 0
    for node in G.nodes():
        nodes.append(counter)
        dict.update({node:counter})
        counter +=1
    output.add_nodes_from(nodes)
    for edge in G.edges():
        output.add_edge(dict[edge[0]],dict[edge[1]])
    return output

def similarity(v1,v2,w1,w2):
    v3=[]
    similarity2=0
    for i in range(len(v1)):
        v3.append(v2[i])
        w3 = w1[:, i]
        w4 = w2[i, :]
        w3= w3.flatten()
        w3= w3.flatten()
        w4= w4[0].flatten()
        w4= w4.flatten()
        w3 = w3[0].tolist()
        w4 = w4[0].tolist()
        M1 = np.array([w3[0], w4[0]])
        similarity1 = cosine_similarity(M1)
        similarity2 = similarity2+similarity1

    similarity2 = similarity2/len(v1)
    M1 = np.array([v1, v3])
    similarity1 = cosine_similarity(M1)
    return similarity1,similarity2

def sort_eigens(v1, w1):
    w2 = []
    v2 = sorted(v1, reverse=True)
    v3 =[]
    for i in range (0,len(v1)):
        v3.append(v2[i])
        x= v2[i]
        b = [item for item in range(len(v1)) if v1[item] == x]
        l = w1[b, :]
        #l= l[0]
        l= l.flatten()
        w2.append(l)
    w2 = np.array(w2)
    return v3, w2

def sort_eigens2(v1, w1):
    w2 = []
    v2 = sorted(v1, reverse=True)
    v3 =[]
    for i in range (0,len(v1)):
        v3.append(v2[i])
        x= v2[i]
        b = [item for item in range(len(v1)) if v1[item] == x]
        l = w1[b, :]
        l= np.array(l)
        l=l[0]
        l = l.tolist()
        w2.append(l)
    w2 = np.matrix(w2)
    #w2 = w2.reverse()
    return v3, w2

def sort_eigens_new(v1, w1,le):
    if le >len(v1):
        le= len(v1)
    w2 = []
    v2 = sorted(v1, reverse=True)
    v3 =[]
    for i in range (0,le):
        v3.append(v2[i])
        x= v2[i]
        b = [item for item in range(len(v1)) if v1[item] == x]
        l = w1[:, b]
        #l = l[0]
        l= l.flatten()
        w2.append(l)
    w2 = np.array(w2)
    return v3, w2

def sort_eigens2_new(v1, w1,le):
    if le >len(v1):
        le= len(v1)
    w2 = []
    v2 = sorted(v1, reverse=True)
    v3 =[]
    for i in range (0,le):
        v3.append(v2[i])
        x= v2[i]
        b = [item for item in range(len(v1)) if v1[item] == x]
        l = w1[:, b]
        l= np.array(l)
        l=l.flatten()
        #l=l[0]
        l = l.tolist()
        w2.append(l)
    w2 = np.matrix(w2)
    #w2 = w2.reverse()
    return v3, w2

def vec_length(v):
    output=0
    for item in v:
        #for item in item1:
            output = output + item.real*item.real +item.imag*item.imag
    return sqrt(output.real)















if __name__ == '__main__':

    datadir = "/../"

    graphNames = [ "karate","dolphins", "squeak", "CA-GrQc", "robots", "fb"]

    compare(graphNames, datadir,2)
    x=1