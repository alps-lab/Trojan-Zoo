# -*- coding: utf-8 -*-


from trojanzoo.plot import *
from trojanzoo.plot.font import *

import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    data = {
        'original': [0.99, 0.9602],
        'adaptive': [0.0986, 0.1668],
    }
    fig = Figure(name='adaptive-strip')
    fig.set_title('')
    color_list = [ting_color['red_carrot'], ting_color['blue_light']]

    size_list = np.array([3, 6])
    width = 0.8
    for i, label in enumerate(list(data.keys())):
        y = np.array(data[label]) * 100
        fig.bar(size_list + (i - 1) * width, y, label=label, width=width, color=color_list[i])

    fig.set_axis_label('x', r'Trigger size ($|\mathit{m}|$)')
    fig.set_axis_label('y', 'TPR (%)')
    fig.set_axis_lim('x', lim=[3, 6], margin=[1.5, 1.5], piece=1, _format='%d')
    fig.set_axis_lim('y', lim=[0, 100], margin=[0, 0], piece=5, _format='%d')
    # fig.set_legend(prop=palatino_bold, loc='upper right')

    fig.save(folder_path="./result/adaptive/")