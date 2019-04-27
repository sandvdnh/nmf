import argparse
import os
import yaml
import numpy as np
from data.data import get_data
from solve import generate_nmf


def main(config, args):
    X, ground_truth = get_data(config)
    W, H, log = generate_nmf(config, X)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='dev2')
    args = parser.parse_args()

    # load config file
    config = yaml.safe_load(open('./config/' + args.config + '.yml'))
    main(config, args)

