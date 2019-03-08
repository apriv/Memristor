import numpy as np


def solveMatrix(a, b):
    c = len(a)
    ret = []
    for x in range(c-1):
        for w in range(1, c-x):
            i = a[x+w][x]/a[x][x]
            for y in range(x, len(a[x])):
                a[x+w][y] -= a[x][y] * i
            b[x+w] -= b[x]*i
        ret.append(0)
    ret.append(0)
    for z in range(c):
        if a[c-1-z][c-1-z] == 0 and not b[c-1-z] == 0:
            return "Error, no solution"
        elif a[c-1-z][c-1-z] == 0 and b[c-1-z] == 0:
            ret[c-1-z] = 0
        else:
            ret[c-1-z] = b[c-1-z]/a[c-1-z][c-1-z]
        if (c-1-z)>0:
            for i in range(c-1-z):
                b[i] -= a[i][c-1-z]*ret[c-1-z]
    return ret

if __name__ == "__main__":
    print("if print array of 0s on the next line of answer, means the answer is right")
    print("regular:")
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
    b = [18, 31, 79]
    c = solveMatrix(a, b)

    b2 = [18, 31, 79]
    a2 = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
    d = np.linalg.solve(a2, b2)


    y = np.dot(a2, c)
    z = np.dot(a2, d)

    print(c)
    b = [18, 31, 79]
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
    for i in range(len(c)):
        for ii in range(len(c)):
            b[ii] -= a[ii][i] * c[i]
    print(b)

    print("multiple solution, only return 1 of them:")
    a= [[1,0],[2,0]]
    b= [2,4]
    c = solveMatrix(a, b)

    a= [[1,0],[2,0]]
    b= [2,4]
    d = np.linalg.solve(a, b)

    print(c)
    a = [[1, 0], [2, 0]]
    b = [2, 4]
    for i in range(len(c)):
        for ii in range(len(c)):
            b[ii] -= a[ii][i] * c[i]
    print(b)

    print("No solution:")
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    b = [18, 31, 79]
    c = solveMatrix(a, b)
    print(c)
    b = [18, 31, 79]
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    if not c == "Error, no solution":
        for i in range(len(c)):
            for ii in range(len(c)):
                b[ii] -= a[ii][i]*c[i]
        print(b)

