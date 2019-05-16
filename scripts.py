import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc
import time
import yaml
from lib.utils import get_data, compute_nmf
from lib.experiment import Experiment
from lib.solver import Solver
from lib.solvers.anls_bpp import ANLSBPP
from lib.solvers.hals import HALS
from lib.solvers.mu import MU
from lib.solvers.sparse_hals import SparseHALS
from lib.solvers.sparse_anls_bpp import SparseANLSBPP
from lib.solvers.sparse_hoyer import SparseHoyer

#rc('text', usetex=True)
#'#1CA4FC',  blue
#blue,  orange,, light blue,  pink
COLORS = ['#1CA4FC', '#FC6554', '#2FE6CF', '#ED62A7']
Y_LABELS = {
        'L0_H': '$\ell_0(H)$',
        'L1_H': '$\ell_1(H)$',
        'rel_error': 'Relative error'
        }
LABELS = {
        'sparse_hals': 'HALS-sparse1',
        'sparse_hals1': 'HALS-sparse2',
        'sparse_l0_hals': 'HALS-sparse3',
        'sparse_anls_bpp': 'ANLS-BPP-sparse'
        }


def generate_synthetic_data(n, m, r, l0):
    '''
    generates synthetic sparse dataset
    '''
    l_list = np.ceil(m * l0).astype(np.int32)
    length = len(l_list)
    W = np.abs(np.random.normal(2, scale=3, size=(n, r, len(l_list))))
    W /= np.linalg.norm(W, axis = 0)

    H = np.zeros((r, m, length))
    X = np.zeros((n, m, length))
    index = 0
    for l in l_list:
        x = np.zeros((r, m))
        for i in range(r):
            nonzero = np.random.randint(m, size=(l,))
            random = np.abs(np.random.normal(size=(m,), scale=10))
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
    config['clip'] = False # otherwise methods diverge?
    experiment_config = yaml.safe_load(open('./experiments/peharz.yml'))
    name = 'peharz'
    solvers = experiment_config['solver_list']
    # generate data
    n = experiment_config['n']
    m = experiment_config['m']
    r = experiment_config['r']
    l0 = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    X, W, H = generate_synthetic_data(n, m, r, l0)
    l0_axis = np.array([Solver.get_nonzeros(H[:, :, i]) for i in range(len(l0))])
    print('Data generated, rank of X: ', np.linalg.matrix_rank(X[:, :, 0]))
    accuracy = np.zeros((len(l0), len(solvers)))
    total = [np.zeros((len(experiment_config['solver_list']), 0)) for feature in experiment_config['plot']]
    for i in range(len(l0)):
        # generate experiment object
        config['project_l0'] = l0_axis[i]
        experiment = Experiment(config, X[:, :, i], experiment_config)
        #print([solver.name for solver in experiment.solvers])
        experiment.run()
        summary = experiment.summary
        #summary = experiment.get_summary()
        for i, feature in enumerate(experiment_config['plot']):
            a = summary[feature]
            a = np.array(a).reshape((len(a), 1))
            total[i] = np.hstack((total[i], a))
    print(total)
    # plotting
    for i, feature in enumerate(experiment_config['plot']):
        fig = plt.figure(figsize=(4, 4))
        ax0 = fig.add_subplot(111)
        #color = ['r', 'g', 'b', 'cyan', 'k']
        ax0.set_xlabel('$\ell_0 (H_o )$')
        for j in range(total[i].shape[0]):
            ax0.plot(l0_axis, total[i][j, :], color=COLORS[j], label = LABELS[solvers[j]], linestyle='--', markersize=15, marker='.')
        ax0.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax0.get_yaxis().set_tick_params(which='both', direction='in')
        ax0.get_xaxis().set_tick_params(which='both', direction='in')
        ax0.grid()
        ax0.set_ylabel(Y_LABELS[feature])
        #ax0.legend()
        #ax0.set_xscale('log')
        #ax0.set_yscale('log')
        s = '_' + str(n) + '_' + str(m) + '_' + str(r)
        fig.savefig('./experiments/' + name + '/' + feature + s + '.pgf', bbox_inches='tight')
        fig.savefig('./experiments/' + name + '/' + feature + s + '.pdf', bbox_inches='tight')


def face_experiment():
    '''
    applies NMF to the faces dataset and generates an image
    '''
    config = yaml.safe_load(open('./config/dev.yml'))
    experiment_config = yaml.safe_load(open('./experiments/face.yml'))
    solvers = experiment_config['solver_list']
    config['dataset'] = 'face'
    r = experiment_config['r']
    config['r'] = r
    name = 'face'
    X, ground_truth = get_data(config)
    #if ground_truth is not -1:
    #    W = ground_truth[0]
    #    H = ground_truth[1]
    print('Data loaded, rank of X: ', np.linalg.matrix_rank(X))
    experiment = Experiment(config, X, experiment_config)
    experiment()
    images = np.zeros((r, 19, 19))
    for solver in experiment.solvers:
        W = solver.solution[0]
        W /= np.max(W, axis=0)
        W = 1 - W
        for i in range(r):
            images[i, :, :] = np.reshape(W[:, i], (19, 19))

        d = 0.05
        plt.subplots_adjust(wspace=d, hspace=d)
        fig, ax = plt.subplots(4, 4)
        fig.set_figheight(8)
        fig.set_figwidth(8)
        for m in range(4):
            for n in range(4):
                ax[m, n].imshow(images[4 * m + n, :, :], cmap='gray', vmin=0, vmax=1)
                ax[m, n].set_xticks([])
                ax[m, n].set_yticks([])
        fig.savefig('./experiments/' + name + '/' + solver.name + '.pgf', bbox_inches='tight')
        fig.savefig('./experiments/' + name + '/' + solver.name + '.pdf', bbox_inches='tight')
    return 0


def complexity_experiment():
    '''
    tries to compare the complexity of iterations 
    '''
    # load experiment config file
    config = yaml.safe_load(open('./config/dev.yml'))
    config['clip'] = False # otherwise methods diverge?
    experiment_config = yaml.safe_load(open('./experiments/complexity.yml'))
    name = 'complexity'
    solvers = experiment_config['solver_list']
    # generate data
    n = np.arange(190, 290, 10)
    m = np.arange(190, 290, 10)
    r = [5, 10, 15]
    l0 = [0.7]
    threshold = 0.2
    iterations = np.zeros((len(r), len(n), len(solvers)))
    for i in range(len(n)):
        for j in range(len(r)):
            X, W, H = generate_synthetic_data(n[i], m[i], r[j], l0)
            print('Data generated, rank of X: ', np.linalg.matrix_rank(X[:, :, 0]))
            experiment = Experiment(config, X[:, :, i], experiment_config)
            experiment.run()
            for k, solver in enumerate(experiment.solvers):
                iterations_ = solver.output['iteration']
                rel_error = solver.output['rel_error']
                index_list = np.where(np.array(rel_error) < threshold)[0]
                if len(index_list) > 0:
                    index = index_list[0]
                    iterations[j, i, k] = iterations_[index]
                else:
                    iterations[j, i, k] = iterations_[-1]

    fig = plt.figure(figsize=(6, 6))
    ax0 = fig.add_subplot(111)
    #color = ['r', 'g', 'b', 'cyan', 'k']
    ax0.set_xlabel('Size of $X$')
    ax0.set_ylabel('Iterations until relative error $< 0.3$')
    for i in range(len(r)):
        for j in range(len(solvers)):
            ax0.plot(n * m, iterations[i, :, j], color=COLORS[j], label = solvers[j], linestyle='--', markersize=15, marker='.')
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax0.get_yaxis().set_tick_params(which='both', direction='in')
    ax0.get_xaxis().set_tick_params(which='both', direction='in')
    ax0.grid()
    #ax0.set_ylabel(Y_LABELS[feature])
    ax0.legend()
    #ax0.set_xscale('log')
    #ax0.set_yscale('log')
    #s = '_' + str(n) + '_' + str(m) + '_' + str(r)
    fig.savefig('./experiments/' + name + '/' + 'graph.pgf', bbox_inches='tight')
    fig.savefig('./experiments/' + name + '/' + 'graph.pdf', bbox_inches='tight')


def classic_experiment():
    '''
    '''
    config = yaml.safe_load(open('./config/dev.yml'))
    config['dataset'] = 'face'
    experiment_config = yaml.safe_load(open('./experiments/classic.yml'))
    name = 'classic'
    solvers = experiment_config['solver_list']
    X, _ = get_data(config)

    experiment = Experiment(config, X, experiment_config)
    experiment()


    fig = plt.figure(figsize=(6, 6))
    ax0 = fig.add_subplot(111)
    ax0.set_xlabel('iteration')
    ax0.set_ylabel('Relative error')

    for i, solver in enumerate(experiment.solvers):
        x_axis = np.array(solver.output['iteration'])
        y_axis = np.array(solver.output['rel_error'])
        ax0.plot(x_axis, y_axis, color=COLORS[i], label = solvers[i], linestyle='--', markersize=8, marker='.')
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax0.get_yaxis().set_tick_params(which='both', direction='in')
    ax0.get_xaxis().set_tick_params(which='both', direction='in')
    ax0.grid()
    ax0.set_ylim([0.1, 0.3])
    ax0.legend()
    #ax0.set_ylabel(Y_LABELS[feature])
    #ax0.set_xscale('log')
    #ax0.set_yscale('log')
    #s = '_' + str(n) + '_' + str(m) + '_' + str(r)
    fig.savefig('./experiments/' + name + '/' + 'graph_iter.pgf', bbox_inches='tight')
    fig.savefig('./experiments/' + name + '/' + 'graph_iter.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(6, 6))
    ax0 = fig.add_subplot(111)
    ax0.set_xlabel('time [s]')
    ax0.set_ylabel('Relative error')
    for i, solver in enumerate(experiment.solvers):
        x_axis = np.array(solver.output['time'])
        y_axis = np.array(solver.output['rel_error'])
        ax0.plot(x_axis, y_axis, color=COLORS[i], label = solvers[i], linestyle='--', markersize=8, marker='.')
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax0.get_yaxis().set_tick_params(which='both', direction='in')
    ax0.get_xaxis().set_tick_params(which='both', direction='in')
    ax0.grid()
    ax0.set_ylim([0.1, 0.3])
    ax0.legend()
    #ax0.set_ylabel(Y_LABELS[feature])
    #ax0.set_xscale('log')
    #ax0.set_yscale('log')
    #s = '_' + str(n) + '_' + str(m) + '_' + str(r)
    fig.savefig('./experiments/' + name + '/' + 'graph_time.pgf', bbox_inches='tight')
    fig.savefig('./experiments/' + name + '/' + 'graph_time.pdf', bbox_inches='tight')



