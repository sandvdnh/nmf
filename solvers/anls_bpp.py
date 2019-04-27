import numpy as np


def divide(F_list, G_list, CTC, CTB):
    '''
    groups similar subproblems together, and extracts corresponding matrices
    returns list of 'problems'. Each entry looks like (CFTC, CFTB, CGTC, CGTB, indices)
    '''
    problems = []
    index_list = []
    index_list.append([0])
    for i in range(1, len(F_list)):
        existed = False
        for j in range(len(index_list)):
            if F_list[index_list[j][0]] == F_list[i]:
                index_list[j].append(i)
                existed = True
        if not existed:
            index_list.append([i])
    for i in range(len(index_list)):
        F = sorted(F_list[index_list[i][0]])
        G = sorted(G_list[index_list[i][0]])
        CFTC = np.zeros((len(F), len(F)))
        CFTB = np.zeros((len(F), CTB.shape[1]))
        CGTC = np.zeros((len(G), len(F)))
        CGTB = np.zeros((len(G), CTB.shape[1]))
        for j in range(len(F)):
            for k in range(len(F)):
                CFTC[j, k] = CTC[F[j], F[k]]
            for k in range(len(G)):
                CGTC[k, j] = CTC[G[k], F[j]]
            CFTB[j, :] = CTB[F[j], :]
        for j in range(len(G)):
            CGTB[j, :] = CTB[G[j], :]
        problems.append((CFTC, CFTB, CGTC, CGTB, index_list[i]))
    return problems

def update(problem, X, Y, F_list, G_list):
    '''
    solves the unconstrained problem for x_F and y_G, and stores the result in X
    '''
    CFTC = problem[0]
    CFTB = problem[1]
    CGTC = problem[2]
    CGTB = problem[3]
    indices = problem[4]
    X_F = np.matmul(np.linalg.inv(CFTC), CFTB)
    Y_G = np.matmul(CGTC, X_F) - CGTB
    q = X.shape[0]
    #X = np.zeros((q, r))
    #Y = np.zeros((q, r))
    F = F_list[indices[0]]
    G = G_list[indices[0]]
    #print('F_list: ', F_list)
    #print('updating rows F: ', F, ' with values ', X_F, ' for columns ', indices)
    #print('updating G: ', G)
    for i in indices:
        X[:, i] = np.zeros(q)
        Y[:, i] = np.zeros(q)
        X[list(F), i] = X_F[:, i]
        Y[list(G), i] = Y_G[:, i]
    stop = np.all(X >= 0) and np.all(Y >= 0) 
    return X, Y, stop 


def nnls(C, B, max_iters):
    '''
    solves the NNLS problem using the block pivoting method.
    '''
    p = C.shape[0]
    q = C.shape[1]
    r = B.shape[1]
    CTC = np.matmul(C.T, C)
    CTB = np.matmul(C.T, B)
    X = np.zeros((q, r))
    Y = -CTB.copy()
    F_list = [set() for i in range(r)]
    G_list = [set(range(q)) for i in range(r)]
    stop = False
    i = 0
    alpha = 3 * np.ones(r)
    beta = (q + 1) * np.ones(r)
    solved = np.array([False for i in range(r)])

    while i < max_iters and not stop:
        # determine V_hat
        #print('BEFORE ANYTHING: ', Y)
        for j in range(r):
            #print(X[j])
            #print(Y[j])
            V = set(np.argwhere(X[:, j] < 0).flat) | set(np.argwhere(Y[:, j] < 0).flat)
            #print('Length of V at start of iteration {}:  '.format(i), len(V))
            #print('with Y: ', Y[:, j])
            #print('with X: ', X[:, j])
            if len(V) > 0:
                if len(V) < beta[j]:
                    beta[j] = len(V)
                    alpha[j] = 3
                    V_hat = V
                elif len(V) >= beta[j] and alpha[j] >=1:
                    alpha[j] -= 1
                    V_hat = V
                elif len(V) >= beta[j] and alpha[j] == 0:
                    #print(beta)
                    #print(len(V))
                    V_hat = {max(V)}
                F = F_list[j]
                G = G_list[j]
                F_new = (F - V_hat) | (V_hat & G)
                G_new = (G - V_hat) | (V_hat & F)
                F_list[j] = F_new.copy()
                G_list[j] = G_new.copy()
                #print('Set F: ', F_new)
            else:
                if not solved[j]:
                    #print('SOLUTION FOUND FOR COLUMN: ', j)
                    solved[j] = True
        # update X
        problems = divide(F_list, G_list, CTC, CTB)
        for problem in problems:
            #print("PROBLEM with indices: ", problem[-1])
            X, Y, stop = update(problem, X, Y, F_list, G_list)
        # check if stop criterion is met
        i += 1
        #print('X and Y: ', X, Y)
    #print('NNLS ran for {} iterations'.format(i))
    return X, i


def solve(config, X):
    '''
    ANLS-BPP procedure
    '''
    iters = config['iters']
    r = config['r']
    nnls_iters = config['nnls_iters']
    eps = config['eps']
    delay = config['delay']
    iters = config['iters']
    n, m = X.shape

    H = np.abs(np.random.normal(loc=0, scale=2, size=(r, m)))
    i = 0
    stop = False
    objective = []
    while not stop:
        if eps > 0:
            if i > delay:
                if np.abs(objective[i - delay] - objective[i - 1]) < eps:
                    stop = True
        else:
            if i >= iters:
                stop = True
        WT, _ = nnls(H.T, X.T, nnls_iters)
        #print(WT)
        W = WT.T
        H, _ = nnls(W, X, nnls_iters)
        #print(H)
        normalization = np.linalg.norm(W, axis=0).reshape((1, 2))
        #print(normalization)
        W /= normalization
        H *= normalization.T

        objective.append(np.linalg.norm(np.matmul(W, H) - X))
        if config['verbose']:
            print('ERROR: ', objective[-1])
        i += 1
    return W, H
