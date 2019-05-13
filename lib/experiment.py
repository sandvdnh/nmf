import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import time
from lib.solver import Solver
from lib.solvers.anls_bpp import ANLSBPP
from lib.solvers.hals import HALS
from lib.solvers.mu import MU
from lib.solvers.sparse_hals import SparseHALS
from lib.solvers.sparse_anls_bpp import SparseANLSBPP
from lib.solvers.sparse_hoyer import SparseHoyer

class Experiment(object):
    '''
    Class used to group different nmf runs on the same dataset together,
    and plot and compare features stored in the output of each solver object.
    '''
    def __init__(self, config, X, experiment_config):
        '''
        it is assumed that the config['log'] entry contains the features in 'features'
        '''
        solver_list = experiment_config['solver_list']
        features = experiment_config['features']
        self.solvers = []
        # Add features to config['log']
        log_set = set(config['log'])
        log_set = log_set | set(features)
        config['log'] = list(log_set)
        for method in solver_list:
            if method == 'anls_bpp':
                solver = ANLSBPP(config, X)
            elif method == 'hals':
                solver = HALS(config, X)
            elif method == 'mu':
                solver = MU(config, X)
            elif method == 'sparse_hals':
                solver = SparseHALS(config, X)
            elif method == 'sparse_anls_bpp':
                solver = SparseANLSBPP(config, X)
            elif method == 'sparse_hoyer':
                solver = SparseHoyer(config, X)
            self.solvers.append(solver)
        self.features = features + ['time', 'iteration']
        self.data = [] # each list in this list corresponds to a feature in self.features

        self.figsize = experiment_config['figsize']
        self.across_time = experiment_config['across_time']
        self.name = experiment_config['name']


    def run(self):
        '''
        Execute all solvers, and store the relevant output in self.data
        '''
        for solver in self.solvers:
            print('Executing ', solver.name, '...')
            solver.solve()
        for feature in self.features[:-2]:
            data_entry = [solver.output[feature] for solver in self.solvers]
            self.data.append(data_entry)

    def get_summary(self):
        '''
        function which returns the best values after all algorithms have finished
        '''
        summary = dict()
        for feature in self.features[:-2]:
            values = [solver.output[feature][-1] for solver in self.solvers]
            summary[feature] = values

        summary['W'] = [solver.solution[0] for solver in self.solvers]
        summary['H'] = [solver.solution[1] for solver in self.solvers]
        return summary


    def _plot_feature(self, feature):
        '''
        Creates plot of a single feature across all solvers
        '''
        fig = plt.figure(figsize=self.figsize)
        ax0 = fig.add_subplot(111)
        color = ['r', 'g', 'b', 'cyan', 'k']

        index = self.features.index(feature)
        data_entry = self.data[index]
        for i, vector in enumerate(data_entry):
            if self.across_time:
                x_axis = self.solvers[i].output['time']
                ax0.set_xlabel('Time')
            else:
                x_axis = self.solvers[i].output['iteration']
                ax0.set_xlabel('iteration')
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


    def __call__(self):
        '''
        Executes all solvers, and generates all features
        '''
        self.run()
        print(self.features)
        for feature in self.features[:-2]:
            self._plot_feature(feature)
