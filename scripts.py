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


def generate_synthetic_data(n, m, k, l0):
    '''
    generates synthetic sparse dataset
    '''
    l_list = np.ceil(m * l0).astype(np.int32)
    length = len(l_list)
    W = np.abs(np.random.normal(size=(n, k, len(l_list))))
    W /= np.linalg.norm(W, axis = 0)

    H = np.zeros((k, m, length))
    X = np.zeros((n, m, length))
    index = 0
    for l in l_list:
        x = np.zeros((k, m))
        for i in range(k):
            nonzero = np.random.randint(m, size=(l,))
            random = np.abs(np.random.normal(size=(m,), scale = 3)) + 2
            x[i, nonzero] = random[nonzero]
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
    l0 = np.array([0.1, 0.2, 0.3, 0.4])
    X, W, H = generate_synthetic_data(n, m, k, l0)
    print('l0 norm H: ', Solver.get_nonzeros(H[:, :, 0]))
    print('l0 norm H: ', Solver.get_nonzeros(H[:, :, 1]))
    print('l0 norm H: ', Solver.get_nonzeros(H[:, :, 2]))
    #print(X)
    accuracy = np.zeros((len(l0), len(solvers)))
    for i in range(1):
        # generate experiment object
        experiment = Experiment(config, X[:, :, i], experiment_config)
        experiment.run()
        summary = experiment.get_summary()
        accuracy[i, :] = np.array(summary['error'])

    # plotting
    fig = plt.figure(figsize=(6, 4))
    ax0 = fig.add_subplot(111)
    color = ['r', 'g', 'b', 'cyan', 'k']
    ax0.set_xlabel('nonzero coefficients')
    ax0.plot(np.array(x_axis), vector, label=self.solvers[i].name, color=color[i])
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax0.get_yaxis().set_tick_params(which='both', direction='in')
    ax0.get_xaxis().set_tick_params(which='both', direction='in')
    ax0.set_ylabel(feature)
    ax0.legend()
    #ax0.set_xscale('log')
    #ax0.set_yscale('log')
    fig.savefig('./experiments/' + self.name + '/' + feature + '.pdf', bbox_inches='tight')



def faces_experiment():
    '''
    applies NMF to the faces dataset and generates an image
    '''
    pass

