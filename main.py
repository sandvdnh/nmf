import argparse
import os
import yaml
import numpy as np
from lib.utils import get_data, compute_nmf
from lib.experiment import Experiment


def main(config, args, experiment_config={}):
    X, ground_truth = get_data(config)
    print('Data loaded')
    #compute_nmf(config, X)
    experiment = Experiment(config, X, experiment_config)
    experiment()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='dev')
    parser.add_argument('--experiment_config', default='test')
    args = parser.parse_args()

    # load config file
    config = yaml.safe_load(open('./config/' + args.config + '.yml'))
    experiment_config = yaml.safe_load(open('./experiments/' + args.experiment_config + '.yml'))
    config['solver'] = 'mu'
    config['dataset'] = 'face'
    main(config, args, experiment_config)

