import numpy as np
import solvers.mu as mu
import solvers.anls_bpp as anls_bpp
import solvers.hals as hals
import solvers.sparse_nmf as sparse_nmf
import solvers.sparse_anls as sparse_anls

def generate_nmf(config, X):
    '''
    executes solver specified in config on data matrix X
    '''
    if config['solver'] == 'mu':
        W, H = mu.solve(config, X)
        log = 0
    if config['solver'] == 'anls_bpp':
        W, H = anls_bpp.solve(config, X)
        log = 0
    if config['solver'] == 'hals':
        W, H = hals.solve(config, X)
        log = 0
    if config['solver'] == 'sparse_nmf':
        W, H = sparse_nmf.solve(config, X)
        log = 0
    if config['solver'] == 'sparse_anls':
        W, H = sparse_anls.solve(config, X)
    return W, H, log
