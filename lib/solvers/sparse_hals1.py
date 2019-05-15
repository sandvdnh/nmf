import numpy as np
from lib.solver import Solver


class SparseHALS1(Solver):
    '''
    Solver subclass that implements the HALS algorithm
    '''
    def __init__(self, config, X):
        Solver.__init__(self, config, X)
        self.name = 'sparse_hals1'
        self.alpha = config['alpha1']
        print('Sparse HALS solver created!')

    def _update_WH(self, W, H):
        A = W
        B = H.T
        W = np.matmul(self.X.T, A)
        V = np.matmul(A.T, A)
        m = B.shape[0]
        l1 = np.linalg.norm(B, axis=0, ord=1)
        #l2 = np.linalg.norm(B, axis=0, ord=2)
        for j in range(self.r):
            vec = B[:, j] + W[:, j] - np.dot(B, V[:, j]) - self.alpha * np.ones(m) * l1[j]
            #vec = B[:, j] + W[:, j] - np.dot(B, V[:, j]) - 2 * self.alpha * (np.ones(m) * l1[j] - B[:, j])
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
        a = 1/2 * np.linalg.norm(np.matmul(W, H) - self.X) ** 2 + self.alpha * np.linalg.norm(np.linalg.norm(H, axis=1, ord=1)) ** 2
        #a = 1/2 * np.linalg.norm(np.matmul(W, H) - self.X) ** 2 + self.alpha * (np.linalg.norm(np.linalg.norm(H, axis=1, ord=1)) ** 2 - np.linalg.norm(np.linalg.norm(H, axis=1, ord=2)) ** 2)
        self.objective.append(a)
