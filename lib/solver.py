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
        elapsed = 0
        self.log(W, H, elapsed, i)
        self._update_objective(W, H)
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
                if not i % self.config['verbose']:
                    print('ITERATION ', i, '  OBJECTIVE: ', self.output['error'][-1])
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

