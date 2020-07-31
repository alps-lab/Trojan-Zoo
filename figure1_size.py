# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='dataset', default='cifar10')
    args = parser.parse_args()
    name = 'figure1 %s alpha' % args.dataset
    fig = Figure(name)
    fig.set_axis_label('x', 'Trigger Size')
    fig.set_axis_label('y', 'Max Re-Mask Accuracy')
    fig.set_axis_lim('x', lim=[0, 100], piece=10, margin=[0, 5],
                     _format='%d')
    fig.set_axis_lim('y', lim=[0, 100], piece=5, margin=[0.0, 5.0],
                     _format='%d')
    fig.set_title(fig.name)

    color_list = [ting_color['red'], ting_color['yellow'], ting_color['green'], ting_color['blue']]

    x = np.linspace(1, 10, 10)**2
    y = {
        'cifar10': {
            'badnet': [67.240, 78.710, 85.060, 88.150, 90.590, 91.550, 93.230],
            'latent_backdoor': [31.890, 97.160, 99.000, 99.790, 100.000, 100.000, 100.000],
            'trojannn': [63.690, 75.360, 82.730, 96.190],
        },
        'gtsrb': {
            'badnet': [42.68, 56.757, 53.134, 59.779, 56.044, 55.03, 61.806, 73.724, 76.107, 79.936],
            'latent_backdoor': [9.572, 92.080, 99.981, 99.944, 100, 100, 100, 100, 100, 100],
            'trojannn': [35.998, 59.816, 57.789, 41.235, 73.048, 94.125, 98.104, 31.306, 93.919, 95.420],
            'targeted_backdoor': [23.161, 33.821, 13.626, 34.835, 18.468, 20.721, 23.742, 38.251, 33.615, 68.731],
        },
    }
    for i, (key, value) in enumerate(y[args.dataset].items()):
        fig.curve(x[:len(value)], value, color=color_list[i], label=key)
        fig.scatter(x[:len(value)], value, color=color_list[i])
    fig.save('./result/')