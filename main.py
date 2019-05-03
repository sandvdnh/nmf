import argparse
import os
import yaml
import numpy as np
from data import get_data
from solve import compute_nmf


def main(config, args):
    X, ground_truth = get_data(config)
    print(X.shape)
    print('Data loaded, computing NMF...')
    compute_nmf(config, X)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='dev')
    args = parser.parse_args()

    # load config file
    config = yaml.safe_load(open('./config/' + args.config + '.yml'))
    config['solver'] = 'mu'
    config['dataset'] = 'face'
    main(config, args)

