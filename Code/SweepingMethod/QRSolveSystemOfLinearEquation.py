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
    return x

A = [[0,3,1,9],[2,0,4,-2],[7,2,1,1],[4,3,9,6]]
a = [[0,3,1,9],[2,0,4,-2],[7,2,1,1],[4,3,9,6]]
B = [6,2,7,9]
b = [6,2,7,9]

x = solve2(A, B, 4)
x2 = np.linalg.solve(a,b)
print(x)
print(x2)