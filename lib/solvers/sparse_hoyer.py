import numpy as np
from lib.solver import Solver


class SparseHoyer(Solver):
    '''
    Solver subclass that implements the sparse NMF algorithm proposed in Hoyer (2004)
    '''
    def __init__(self, config, X):
        Solver.__init__(self, config, X)
        self.name = 'sparse_hoyer'
        self.sparsity = config['sparsity']
        self.mu = config['mu']
        print('SparseHoyer solver created!')

    def _update_WH(self, W, H):
        '''
        updates W and H in each iteration
        '''
        WH = np.matmul(W, H)
        WHHT = np.matmul(WH, H.T)
        VHT = np.matmul(self.X, H.T)
        W = W * VHT / WHHT

        WH = np.matmul(W, H)
        grad = np.matmul(W.T, WH - self.X)
        while 1:
            H -= self.mu * grad
            H = self.project(H)

            objective = np.linalg.norm(np.matmul(W, H) - self.X)
            if self.objective[-1] < objective and mu > 1e-50:
                self.mu /= 2
            elif self.objective[-1] > objective:
                break
            else:
                print('CONVERGED...')
        return W, H

    def _update_objective(self, W, H):
        '''
        calculates the value of the objective function
        '''
        a = 1/2 * np.linalg.norm(np.matmul(W, H) - self.X) ** 2
        self.objective.append(a)
        pass

    def init_WH(self):
        '''
        OVERRIDES method in Solver class
        '''
        W = np.abs(np.random.normal(loc=0, scale=2, size=(self.n, self.r)))
        WTW = np.matmul(W.T, W)
        H = np.matmul(np.matmul(np.linalg.inv(WTW), W.T), self.X)
        H = H.clip(min=0)
        HHT = np.matmul(H, H.T)
        W = np.transpose(np.matmul(np.matmul(np.linalg.inv(HHT), H), self.X.T))
        W = W.clip(min=0)
        W /= np.linalg.norm(W, axis=0)
        H = self.project(H)
        return W, H

    def project(self, M):
        '''
        projects each column of matrix M to the nearest vector with unchanged L2 norm
        but L1 norm set such that its sparsity equals self.sparsity
        '''
        n = M.shape[0]
        l1_ = lambda x: x * (np.sqrt(n) - (np.sqrt(n) - 1) * self.sparsity)
        for i in range(M.shape[1]):
            vect = M[:, i]
            l2 = np.linalg.norm(vect)
            l1 = l1_(l2)
            M[:, i] = SparseHoyer._project(vect, l1, l2)
            #print('Sparsity level of column: ', SparseHoyer._get_sparsity(M[:, i]))
        return M


    def _project(x, l1, l2):
        '''
        projects a positive vector x onto a vector y with given l1 norm such that
        the distance between x and y is minimal (in the l2 sense)
        '''
        s = x + (l1 - np.sum(x))/len(x)
        Z = set()
        all_ = set(range(len(x)))
        i = 0
        while i <= 3 * len(x):
            m = np.zeros(len(x))
            m[list(all_ - Z)] = l1 / (len(x) - len(Z))
            alpha = SparseHoyer._get_alpha(s, m, l2)
            s = m + alpha * (s - m)
            if np.all(s >= -1e-10):
                s = np.maximum(s, 0)
                return s
            new = np.argwhere(s < 0).flat
            Z = Z | set(new)
            s[list(Z)] = 0
            c = (np.sum(s) - l1) / (len(x) - len(Z))
            s -= c
            i += 1
            #stop = (np.abs(SparseHoyer._get_sparsity(s) - self.sparsity)
        print('PROJECTION UNSUCCESSFUL')
        return s

    def _get_alpha(s, m, l2):
        '''
        solves the quadratic equation for alpha
        '''
        a = np.sum((s - m) * (s - m))
        b = 2 * np.sum(m * (s - m))
        c = np.sum(m * m) - l2 ** 2
        alpha = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        return alpha

    def _get_sparsity(v):
        '''
        problem
        '''
        l2 = np.linalg.norm(v)
        l1 = np.linalg.norm(v, ord=1)
        n = len(v)
        return (np.sqrt(n) - l1/l2) / (np.sqrt(n) - 1)
