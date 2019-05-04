import numpy as np
import time
from lib.solver import Solver


class HALS(Solver):
    '''
    Solver subclass that implements the HALS algorithm
    '''
    def __init__(self, config, X):
        Solver.__init__(self, config, X)
        self.name = 'hals'
        print('HALS object created!')

    def _update_WH(self, W, H):
        A = W
        B = H.T
        W = np.matmul(self.X.T, A)
        V = np.matmul(A.T, A)
        for j in range(self.r):
            vec = B[:, j] + W[:, j] - np.dot(B, V[:, j])
            B[:, j] = np.maximum(vec, 1e-16)
            #if i > 10:
            #    B[:, j] = B[:, j].clip(min=1e-10)
        P = np.matmul(self.X, B)
        Q = np.matmul(B.T, B)
        for j in range(self.r):
            vec = A[:, j] * Q[j, j] + P[:, j] - np.dot(A, Q[:, j])
            A[:, j] = np.maximum(vec, 1e-10)
            A[:, j] /= np.linalg.norm(A[:, j])
            #if i > 10:
            #    A[:, j] = A[:, j].clip(min=1e-10)
        #A = A / np.linalg.norm(A, axis=0)
        return A, B.T


    def _update_objective(self, W, H):
        '''
        calculates the value of the objective function
        '''
        a = np.linalg.norm(np.matmul(W, H) - self.X)
        self.objective.append(a)

#def solve(config, X):
#    '''
#    HALS procedure
#    '''
#    print('RUNNING HALS...')
#    iters = config['iters']
#    r = config['r']
#    #nnls_iters = config['nnls_iters']
#    eps = config['eps']
#    delay = config['delay']
#    iters = config['iters']
#    n, m = X.shape
#
#    i = 0
#    stop = False
#    objective = []
#    elapsed = []
#    l1_norm = []
#
#    # initialize using ALS
#    A, B = init_(config, X)
#    start = time.time()
#    while not stop:
#        if eps > 0:
#            if i > delay:
#                if np.abs(objective[i - delay] - objective[i - 1]) < eps:
#                    stop = True
#        else:
#            if i >= iters:
#                stop = True
#
#        if False:
#            A_new = np.zeros(A.shape)
#            B_new = np.zeros(B.shape)
#            W = np.matmul(X.T, A)
#            V = np.matmul(A.T, A)
#            for j in range(r):
#                B_new[:, j] = B[:, j] + W[:, j] - np.dot(B, V[:, j])
#            if i > 10:
#                B_new = B_new.clip(min=1e-10)
#            B = B_new
#            P = np.matmul(X, B)
#            Q = np.matmul(B.T, B)
#            for j in range(r):
#                A_new[:, j] = A[:, j] * Q[j, j] + P[:, j] - np.dot(A, Q[:, j])
#            #print(A_new)
#            if i > 10:
#                A_new = A_new.clip(min=1e-10)
#            A = A_new / np.linalg.norm(A_new, axis=0)
#        else:
#            W = np.matmul(X.T, A)
#            V = np.matmul(A.T, A)
#            for j in range(r):
#                vec = B[:, j] + W[:, j] - np.dot(B, V[:, j])
#                B[:, j] = np.maximum(vec, 1e-16)
#                #if i > 10:
#                #    B[:, j] = B[:, j].clip(min=1e-10)
#            P = np.matmul(X, B)
#            Q = np.matmul(B.T, B)
#            for j in range(r):
#                vec = A[:, j] * Q[j, j] + P[:, j] - np.dot(A, Q[:, j])
#                A[:, j] = np.maximum(vec, 1e-10)
#                A[:, j] /= np.linalg.norm(A[:, j])
#                #if i > 10:
#                #    A[:, j] = A[:, j].clip(min=1e-10)
#            #A = A / np.linalg.norm(A, axis=0)
#        objective.append(np.linalg.norm(np.matmul(A, B.T) - X))
#        elapsed.append(time.time() - start)
#
#        if config['verbose']:
#            print('RESIDUAL: ', objective[-1])
#        i += 1
#    output = {
#            'objective': objective,
#            'time': elapsed,
#            'iterations': i}
#    return A, B.T, output
