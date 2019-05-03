import numpy as np
import os
import math
from .solver import Solver

class MU(Solver):
    '''
    Solver subclass that implements the MU algorithm
    '''
    def __init__(self, config, X):
        Solver.__init__(self, config, X)
        print('MU solver created!')

    def _update_WH(self, W, H):
        '''
        updates W and H
        '''
        w = W
        h = H
        v = self.X
        wTv = np.matmul(w.T, v)
        wTwh = np.matmul(np.matmul(w.T, w), h)
        h = h * wTv / wTwh
        vhT = np.matmul(v, h.T)
        whhT = np.matmul(w, np.matmul(h, h.T))
        w = w * vhT / whhT
        return w, h

    def _update_objective(self, W, H):
        '''
        calculates the value of the objective function
        '''
        a = np.linalg.norm(np.matmul(W, H) - self.X)
        self.objective.append(a)

#def solve(config, X):
#    '''
#    Calculates a rank r approximation to a nonnegative matrix v = w_ * h_
#    using the multiplicative update rules described in Lee/Seung
#    '''
#    # initialize w and h to random positive matrices
#    iters = config['iters']
#    r = config['r']
#    v = X
#    n, m = v.shape
#    w = np.abs(np.random.normal(size=(n, r), scale=1))
#    h = np.abs(np.random.normal(size=(r, m), scale=1))
#
#    # iterate
#    for i in range(iters):
#        wTv = np.matmul(w.T, v)
#        wTwh = np.matmul(np.matmul(w.T, w), h)
#        h = h * wTv / wTwh
#        vhT = np.matmul(v, h.T)
#        whhT = np.matmul(w, np.matmul(h, h.T))
#        w = w * vhT / whhT
#        if config['verbose']:
#            if i % 30 == 0:
#                print('error: ', np.linalg.norm(v - np.matmul(w, h)))
#    return w, h


#if __name__ == '__main__':
#    n, m, r = 100, 40, 10
#    iters = 10000
#    v, w, h = get_vwh_(n, m, r)
#    print('rank: ', np.linalg.matrix_rank(v))
#    multiplicative_update(v, 10, w, h, iters)
