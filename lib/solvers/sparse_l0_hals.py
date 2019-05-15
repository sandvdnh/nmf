import numpy as np
from lib.solver import Solver


class SparseL0HALS(Solver):
    '''
    Solver subclass that implements the HALS algorithm while projecting to a given l_0 norm in each iteration
    '''
    def __init__(self, config, X):
        Solver.__init__(self, config, X)
        self.name = 'l0_projection'
        self.l0 = config['project_l0']
        print('SparseL0 solver created!')

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
        B = self._project_to_l0(B)
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

    def _project_to_l0(self, H):
        '''
        projects matrix H to given l0 norm
        '''
        shape = H.shape
        n = np.prod(H.shape)
        to_zero = int(np.ceil((Solver.get_nonzeros(H) - self.l0) * n))
        if to_zero > 0:
            vec = H.flatten()
            vec_argsort = vec.argsort()
            vec1 = vec.copy()
            vec.sort()
            nonzero_ = np.nonzero(vec > 1e-12)[0][0]
            first_nonzero = vec_argsort[np.nonzero(vec > 1e-12)[0][0]]
            indices = vec_argsort[nonzero_:nonzero_ + to_zero]
            vec1[indices] = 0
            H = np.reshape(vec1, shape)
        return H


