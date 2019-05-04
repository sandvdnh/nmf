import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import time
from abc import ABCMeta, abstractmethod


class Solver(object, metaclass=ABCMeta):
    '''
    Abstract class for any algorithm solving the NMF problem.
    '''
    def __init__(self, config, X):
        self.config = config
        self.X = X
        self.n = X.shape[0]
        self.m = X.shape[1]
        self.r = config['r']
        self.objective = []

        # initialize output dictionary
        self.output = {}
        for key in self.config['log']:
            print(key)
            self.output[key] = []


    def solve(self):
        '''
        Solves the NMF problem
        '''
        iters = self.config['iters']
        r = self.config['r']
        eps = self.config['eps']
        delay = self.config['delay']
        iters = self.config['iters']

        # initialize H
        W, H = self.init_WH()
        stop = False
        start = time.time()
        i = 0
        while not stop:
            if eps > 0:
                if i > delay:
                    if np.abs(self.objective[i - delay] - self.objective[i - 1]) < eps:
                        stop = True
            else:
                if i >= iters:
                    stop = True

            W, H = self._update_WH(W, H)
            self._update_objective(W, H)
            elapsed = time.time() - start
            self.log(W, H, elapsed, i)
            if self.config['verbose']:
                print(self.output['error'][-1])
            i += 1

        self.solution = (W, H)
        self.output['objective'] = self.objective


    @abstractmethod
    def _update_WH(self):
        '''
        updates W, H in each iteration
        '''
        pass


    @abstractmethod
    def _update_objective(self, W, H):
        '''
        calculates value of the objective function with the current values of W and H
        '''
        pass


    def init_WH(self):
        '''
        returns initialized W and H matrices based on the ALS method.
        '''
        W = np.abs(np.random.normal(loc=0, scale=2, size=(self.n, self.r)))
        WTW = np.matmul(W.T, W)
        H = np.matmul(np.matmul(np.linalg.inv(WTW), W.T), self.X)
        H = H.clip(min=0)
        HHT = np.matmul(H, H.T)
        W = np.transpose(np.matmul(np.matmul(np.linalg.inv(HHT), H), self.X.T))
        W = W.clip(min=0)
        W /= np.linalg.norm(W, axis=0)
        return W, H


    def log(self, W, H, elapsed, i):
        '''
        logs properties listed in config['logger']
        values is a tuple containing:
        - W
        - H
        - X
        - elapsed
        - objective
        - i
        By default, the value of the objective function, the elapsed time per iteration,
        and the iteration number are logged.
        Returns the modified output dictionary
        '''
        self.output['time'].append(elapsed)
        error = np.linalg.norm(np.matmul(W, H) - self.X)
        self.output['error'].append(error)
        self.output['iteration'].append(i)
        list_ = self.config['log']
        if 'L1_W' in list_:
            self.output['L1_W'].append(np.linalg.norm(W, ord=1))
        if 'L1_H' in list_:
            self.output['L1_H'].append(np.linalg.norm(H, ord=1))
        return 0


#class Experiment(object):
#    '''
#    Class used to group different nmf runs on the same dataset together,
#    and plot and compare features stored in the output of each solver object.
#    '''
#    def __init__(self, config, X, experiment_config):
#        '''
#        it is assumed that the config['log'] entry contains the features in 'features'
#        '''
#        solver_list = experiment_config['solver_list']
#        features = experiment_config['features']
#        self.solvers = []
#        for method in solver_list:
#            if method == 'anls_bpp':
#                solver = ANLSBPP(config, X)
#            elif method == 'hals':
#                solver = HALS(config, X)
#            elif method == 'mu':
#                solver = MU(config, X)
#            self.solvers.append(solver)
#        self.features = features.append('time').append('iteration')
#        self.data = [] # each list in this list corresponds to a feature in self.features
#
#        self.figsize = experiment_config['figsize']
#        self.across_time = experiment_config['across_time']
#        self.name = experiment_config['name']
#
#
#    def run(self):
#        '''
#        Execute all solvers, and store the relevant output in self.data
#        '''
#        for solver in self.solvers:
#            print('Executing ', solver.name, '...')
#            solver.solve()
#        for feature in features[:-2]:
#            data_entry = [solver.output[feature] for solver in self.solvers]
#            self.data.append(data_entry)
#
#
#    def _plot_feature(self, feature):
#        '''
#        Creates plot of a single feature across all solvers
#        '''
#        fig = plt.figure(figsize=self.figsize)
#        ax0 = fig.add_subplot(111)
#        color = ['r', 'g', 'b', 'cyan', 'k']
#
#        index = self.features.index(feature)
#        data_entry = self.data[index]
#        for i, vector in enumerate(data_entry):
#            if self.across_time:
#                x_axis = self.solvers[i].output['time']
#                ax0.set_xlabel('Time')
#            else:
#                x_axis = self.solvers[i].output['iteration']
#                ax0.set_xlabel('iteration')
#            ax0.plot(x_axis, vector, label=self.solvers[i].name, color=color[i])
#        ax0.yaxis.set_major_formatter(FormatStrFormatter('%g'))
#        ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
#        ax0.get_yaxis().set_tick_params(which='both', direction='in')
#        ax0.get_xaxis().set_tick_params(which='both', direction='in')
#        ax0.set_ylabel(feature)
#        fig.savefig('./experiments/' + self.name + '/' + feature + '.pdf', bbox_inches='tight')
#
#
#    def __call__(self):
#        '''
#        Executes all solvers, and generates all features
#        '''
#        self.run()
#        for feature in self.features[:-2]:
#            self._plot_feature(self, feature)


