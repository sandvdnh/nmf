import numpy as np
import os
import math
#import matplotlib.pyplot as plt


#def get_vwh_(n, m, r):
#    '''
#    returns random positive matrices v, w, h such that v = w * h and:
#    - v has shape (n, m)
#    - w has shape (n, r)
#    - h has shape (r, m)
#    '''
#    w = np.abs(np.random.normal(size=(n, r), scale=1))
#    h = np.abs(np.random.normal(size=(r, m), scale=1))
#    v = np.matmul(w, h)
#    return v, w, h


#def multiplicative_update_(v, r, w_, h_, iters):
#    '''
#    Calculates a rank r approximation to a nonnegative matrix v = w_ * h_
#    using the multiplicative update rules described in Lee/Seung
#    '''
#    # initialize w and h to random positive matrices
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
#        if i % 30 == 0:
#            print('error: ', np.linalg.norm(v - np.matmul(w, h)))
#    return w, h


def solve(config, X):
    '''
    Calculates a rank r approximation to a nonnegative matrix v = w_ * h_
    using the multiplicative update rules described in Lee/Seung
    '''
    # initialize w and h to random positive matrices
    iters = config['iters']
    r = config['r']
    v = X
    n, m = v.shape
    w = np.abs(np.random.normal(size=(n, r), scale=1))
    h = np.abs(np.random.normal(size=(r, m), scale=1))

    # iterate
    for i in range(iters):
        wTv = np.matmul(w.T, v)
        wTwh = np.matmul(np.matmul(w.T, w), h)
        h = h * wTv / wTwh
        vhT = np.matmul(v, h.T)
        whhT = np.matmul(w, np.matmul(h, h.T))
        w = w * vhT / whhT
        if config['verbose']:
            if i % 30 == 0:
                print('error: ', np.linalg.norm(v - np.matmul(w, h)))
    return w, h


#if __name__ == '__main__':
#    n, m, r = 100, 40, 10
#    iters = 10000
#    v, w, h = get_vwh_(n, m, r)
#    print('rank: ', np.linalg.matrix_rank(v))
#    multiplicative_update(v, 10, w, h, iters)
