import numpy as np


def project(config, x, l1, l2):
    '''
    projects a vector x onto a vector y with given l1 norm such that
    the distance between x and y is minimal (in the l2 sense)
    '''
    s = np.zeros(len(x))
    s = x + (l1 - np.sum(x))/len(x)
    Z = set()
    for i in range(len(x)):
        m = np.zeros(len(x))
        m[list(Z)] = l1 / (len(x) - len(Z))
        alpha = get_alpha(s, m, l2)
        s = m + alpha * (s - m)
        print(np.linalg.norm(s, ord=1))
        if np.all(s >= 0):
            return s
        new = np.argwhere(s < 0).flat
        Z = Z | new
        s[list(Z)] = 0
        c = (np.sum(s) - l1) / (len(x) - len(Z))
        s -= c
    return s

def get_alpha(s, m, l2):
    '''
    solves the quadratic equation for alpha
    '''
    a = np.sum((s - m) * (s - m))
    b = 2 * np.sum(m * (s - m))
    c = np.sum(m * m) - l2 ** 2
    alpha = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return alpha

def solve(config, X):
    '''
    Projected gradient descent with sparseness (Hoyer et al. 2004)
    '''
    iters = config['iters']
    r = config['r']
    #nnls_iters = config['nnls_iters']
    eps = config['eps']
    delay = config['delay']
    iters = config['iters']
    n, m = X.shape

    i = 0
    stop = False
    objective = []
    x = np.ones(n)

    print(project(config, x, 0.5, 4))
    # initialize using ALS
    #A, B = init_(config, X)
    #while not stop:
    #    if eps > 0:
    #        if i > delay:
    #            if np.abs(objective[i - delay] - objective[i - 1]) < eps:
    #                stop = True
    #    else:
    #        if i >= iters:
    #            stop = True
    #    A_new = np.zeros(A.shape)
    #    B_new = np.zeros(B.shape)
    #    W = np.matmul(X.T, A)
    #    V = np.matmul(A.T, A)
    #    for j in range(r):
    #        B_new[:, j] = B[:, j] + W[:, j] - np.dot(B, V[:, j])
    #    if i > 10:
    #        B_new = B_new.clip(min=1e-6)
    #    B = B_new
    #    P = np.matmul(X, B)
    #    Q = np.matmul(B.T, B)
    #    for j in range(r):
    #        A_new[:, j] = A[:, j] * Q[j, j] + P[:, j] - np.dot(A, Q[:, j])
    #    #print(A_new)
    #    if i > 10:
    #        A_new = A_new.clip(min=1e-6)
    #    A = A_new / np.linalg.norm(A_new, axis=0)

    #    objective.append(np.linalg.norm(np.matmul(A, B.T) - X))
    #    if config['verbose']:
    #        print('RESIDUAL: ', objective[-1])
    #    i += 1
    return A, B.T
