import argparse
import os
import yaml
import numpy as np
from lib.utils import get_data, compute_nmf
from lib.experiment import Experiment
from lib.solver import Solver
from lib.solvers.sparse_l0_hals import SparseL0HALS
from scripts import peharz_experiment


def main(config, args, experiment_config={}):
    #X, ground_truth = get_data(config)
    #if ground_truth is not -1:
    #    W = ground_truth[0]
    #    H = ground_truth[1]
    #print('Data loaded, rank of X: ', np.linalg.matrix_rank(X))
    #compute_nmf(config, X)

    #a = np.identity(6)
    ##print(Solver.get_nonzeros(a))
    #l0 = 0.1
    #H = a
    #shape = H.shape
    #n = np.prod(H.shape)
    #to_zero = int(np.ceil((Solver.get_nonzeros(H) * n - l0 * n)))
    #print(to_zero)
    #if to_zero > 0:
    #    vec = H.flatten()
    #    print(len(vec))
    #    indices = vec.copy().argsort()[:to_zero]
    #    print(indices)
    #    print(len(indices))
    #    vec[indices] = 0
    #    H = np.reshape(vec, shape)
    #print(vec)
    #print(H)
    #print(int(np.ceil((Solver.get_nonzeros(H) - l0) * n)))
    #experiment = Experiment(config, X, experiment_config)
    #experiment()

    peharz_experiment()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='dev')
    parser.add_argument('--experiment_config', default='test')
    args = parser.parse_args()

    # load config file
    config = yaml.safe_load(open('./config/' + args.config + '.yml'))
    experiment_config = yaml.safe_load(open('./experiments/' + args.experiment_config + '.yml'))
    config['solver'] = 'sparse_hoyer'
    config['dataset'] = 'face'
    main(config, args, experiment_config)

