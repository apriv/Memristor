# fixed the problem of taking abs value
#  power iter modified from http://mlwiki.org/index.php/Power_Iteration
import numpy as np

# stop iterating when the value difference is less than max_err
max_err = 0.01


# used for power_iteration
def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)


# to compute one value and vector
def power_iteration(A):

    v = np.random.rand(A.shape[1])
    ev = eigenvalue(A, v)

    while True:
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)

        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < max_err:
            break

        v = v_new  # vector
        ev = ev_new  # value

    return ev_new, v_new


def shifted_power_iteration(A, num_eigs):
    vs = []
    ws = []

    for _ in range(num_eigs):
        w, v = power_iteration(A)
        vs.append(w)
        ws.append(v)
        v = np.array([v])
        B = np.dot(np.transpose(v), v)

        v_norm = np.linalg.norm(v)
        c= w / (v_norm*v_norm)
        C = c*B
        A = A - C
    return vs, ws


if __name__ == '__main__':
    # find leading ev
    (value, vector) = (power_iteration(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])   ))
    print(value)

    values, vectors = shifted_power_iteration(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])  , 3)
    print(values)

    #ev, v = shifted_power_iteration(np.array([[4, 5], [6, 5]]),  2)
    #print(ev,v)