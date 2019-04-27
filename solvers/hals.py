import numpy as np



def solve(config, X):
    '''
    HALS procedure
    '''
    iters = config['iters']
    r = config['r']
    n, m = X.shape

    H = np.abs(np.random.normal(loc=0, scale=2, size=(r, m)))
    i = 0
    stop = False
    nnls_iters = 10
    while i < iters and not stop:
        print('Iteration ', i)
        WT, _ = nnls(H.T, X.T, nnls_iters)
        #print(WT)
        W = WT.T
        H, _ = nnls(W, X, nnls_iters)
        #print(H)
        normalization = np.linalg.norm(W, axis=0).reshape((1, 2))
        #print(normalization)
        W /= normalization
        H *= normalization.T
        i += 1
        if config['verbose']:
            print('ERROR: ', np.linalg.norm(np.matmul(W, H) - X))
    return W, H

