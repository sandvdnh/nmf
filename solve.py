import numpy as np
import solvers.mu as mu
import solvers.anls_bpp as anls_bpp
import solvers.hals as hals

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
    return W, H, log
