import numpy as np
from math import *

def solve2(A, b, n):
    A = np.copy(A)
    A = A.astype(float)
    for i in range(n-1):
        # a is the i's row of A of current state, start from i's column
        # a should be shorter and shorter for each iteration
        a = np.copy(A[:,i])
        a = a[i:]
        # print(a)
        av = 0.0
        for i2 in range(len(a)):
            av += a[i2]*a[i2]
        av = sqrt(av)
        a[0] -= av
        wv = 0
        for i2 in a:
            wv += i2*i2
        H = np.zeros_like(A, dtype="float64")
        # use numpy to create new matrix
        for i2 in range(n):
            H[i2][i2] = 1
        a = a.reshape(1,n-i)
        # 1D array to 2D Array
        a = np.dot(a.T,a)*(2/wv)
        for i2 in range(n-i):
            for i3 in range(n-i):
                H[i2+i][i3+i] -= a[i2][i3]
        b = np.dot(H,b)
        A = np.dot(H,A)
        # use numpy to multiply matrix
        # Hopefully, time = 2^2+3^2+...+n^2 which <n^3 but >n^2
    # R = Hn*Hn-1*...*H2*H1*A
    # Q = QT
    # So, QTb = Hn*...*H1*b
    x = np.zeros_like(b, dtype="float64")
    # use numpy to create new matrix
    for i in range(n):
        x[n-i-1] = b[n-i-1]/A[n-i-1][n-i-1]
        for i2 in range(n-i-1):
            b[i2] -= A[i2][n-i-1]*x[n-i-1]
    return x.tolist()

def QR(A, n):
    R = np.copy(A)
    R = R.astype(float)
    H = []
    for i in range(n-1):
        a = np.copy(R[:,i])
        a = a[i:]
        av = 0.0
        for i2 in range(len(a)):
            av += a[i2]*a[i2]
        av = sqrt(av)
        a[0] -= av
        wv = 0
        for i2 in a:
            wv += i2*i2
        H.append(np.zeros_like(R, dtype="float64"))
        for i2 in range(n):
            H[i][i2][i2] = 1
        a = a.reshape(1,n-i)
        a = np.dot(a.T,a)*(2/wv)
        for i2 in range(n-i):
            for i3 in range(n-i):
                H[i][i2+i][i3+i] -= a[i2][i3]
        R = np.dot(H[i],R)
    Q = H[n-2]
    for i in range(n-2):
        Q = np.dot(H[n-3-i],Q)
    return Q,R

def eigen(A, n):
    R = np.copy(A)
    R = R.astype(float)
    H = []
    B = np.copy(A)
    B = B.astype(float)
    for i in range(n - 2):
        a = np.copy(R[:, i])
        a = a[i+1:]
        av = 0.0
        for i2 in range(len(a)):
            av += a[i2] * a[i2]
        av = sqrt(av)
        a[0] -= av
        wv = 0
        for i2 in a:
            wv += i2 * i2
        H.append(np.zeros_like(R, dtype="float64"))
        for i2 in range(n):
            H[i][i2][i2] = 1
        a = a.reshape(1, n - i-1)
        a = np.dot(a.T, a) * (2 / wv)
        for i2 in range(n - i-1):
            for i3 in range(n - i-1):
                H[i][i2 + i+1][i3 + i+1] -= a[i2][i3]
        R = np.dot(H[i], R)
        B = np.dot(H[i],B)
        B = np.dot(B, H[i])
        # get heisenberg form
    for i in range(45):
        Q, R = QR(B, n)
        B = np.dot(R,Q)
    # approaching
    retl = []
    for i in range(n):
        retl.append(B[i][i])
    retx = []
    for i in retl:
        B = np.copy(A)
        B = B.astype(float)
        for i2 in range(n):
            B[i2][i2] = B[i2][i2]-i
        b = np.copy(B[:, 0])*-1
        b = b[range(1, n)]
        p = range(1, n)
        B = B[p]
        B = B[:, p]
        B[0][0] += 0.0000001
        x = solve2(B, b, n-1)
        x.insert(0,1)
        retx.append(x)
    return retl, retx

A = [[3, 8, 9, 7],[6, 4, -1, 8],[9,1,7,4],[3,5,6,7]]
A = np.random.rand(5,5)
a = np.copy(A)
b,c = eigen(A, 5)
B,C = np.linalg.eig(a)
print(b)
print(B)
for i in range(5):
    print("\n")
    B = b[i]
    C = c[i]
    print(B*np.array(C))
    print(np.dot(A, C))
    print("\n")

