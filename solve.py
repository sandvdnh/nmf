import numpy as np
#import solvers.mu as mu
#import solvers.anls_bpp as anls_bpp
#import solvers.hals as hals
#import solvers.sparse_nmf as sparse_nmf
#import solvers.sparse_anls_bpp as sparse_anls_bpp
#from abc import ABCMeta, abstractmethod
from solvers2.anls_bpp import ANLSBPP
from solvers2.hals import HALS
from solvers2.mu import MU


#def generate_nmf(config, X):
#    '''
#    executes solver specified in config on data matrix X
#    '''
#    if config['solver'] == 'mu':
#        W, H = mu.solve(config, X, output)
#        output = 0
#    if config['solver'] == 'anls_bpp':
#        W, H = anls_bpp.solve(config, X)
#        output = 0
#    if config['solver'] == 'hals':
#        W, H, output = hals.solve(config, X)
#    if config['solver'] == 'sparse_nmf':
#        W, H = sparse_nmf.solve(config, X)
#        output = 0
#    if config['solver'] == 'sparse_anls_bpp':
#        W, H, output = sparse_anls_bpp.solve(config, X)
#        output = 0
#    return W, H, output


def compute_nmf(config, X):
    '''
    Generates Solver object from correct subclass, and computes the NMF
    '''
    if config['solver'] == 'anls_bpp':
        solver = ANLSBPP(config, X)
        solver.solve()
    elif config['solver'] == 'hals':
        solver = HALS(config, X)
        solver.solve()
    elif config['solver'] == 'mu':
        solver = MU(config, X)
        solver.solve()
    return 0

