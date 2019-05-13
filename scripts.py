import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import time
import yaml
from lib.experiment import Experiment
from lib.solver import Solver
from lib.solvers.anls_bpp import ANLSBPP
from lib.solvers.hals import HALS
from lib.solvers.mu import MU
from lib.solvers.sparse_hals import SparseHALS
from lib.solvers.sparse_anls_bpp import SparseANLSBPP
from lib.solvers.sparse_hoyer import SparseHoyer


def generate_synthetic_data(n, m, k):
    '''
    generates synthetic sparse dataset
    '''
    l_list = 300 * np.arange(1, 2)
    length = len(l_list)
    W = np.abs(np.random.normal(size=(n, k, len(l_list))))
    W /= np.linalg.norm(W, axis = 0)
    #W.append(a.copy())

    H = np.zeros((k, m, length))
    X = np.zeros((n, m, length))
    index = 0
    for l in l_list:
        x = np.zeros((k, m))
        for i in range(k):
            nonzero = np.random.randint(m, size=(l,))
            random = np.abs(np.random.normal(size=(m,), scale = 3))
            x[i, nonzero] = random[nonzero]
        #nonzeros = np.round(k * n * l / 100).astype(np.int32)
        #mask = np.random.binomial(1, l / (k * m), size=(k, m)).astype(np.float32)
        #print(l / k * m)
        #mask *= np.abs(np.random.normal(size=mask.shape, scale = 3))
        H[:, :, index] = x.copy()
        X[:, :, index] = np.matmul(W[:, :, index], H[:, :, index])
        index += 1
    return X, W, H


def peharz_experiment():
    '''
    runs the peharz experiment and generates its plots
    '''
    # load experiment config file
    config = yaml.safe_load(open('./config/dev.yml'))
    experiment_config = yaml.safe_load(open('./experiments/peharz.yml'))
    solvers = experiment_config['solver_list']
    # generate data
    n = 300
    m = 200
    k = 9
    X, W, H = generate_synthetic_data(n, m, k)
    #print(X)
    for i in range(1):
        # generate experiment object
        experiment = Experiment(config, X[:, :, i], experiment_config)
        experiment.run()
        summary = experiment.get_summary()

    # plotting



def faces_experiment():
    '''
    applies NMF to the faces dataset and generates an image
    '''

