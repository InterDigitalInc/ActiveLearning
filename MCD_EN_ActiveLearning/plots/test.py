import matplotlib.pyplot as plt
# import mpld3
from matplotlib import rc
import numpy as np
import pandas as pd
import os
import pdb
import sys

sys.path.append('/home/vineeth/Documents/GitWorkSpace/BayesActiveLearning_V2/plots')

plot_pth = 'McDropoutVsBayesianVariationRatio/'
class getMetric:

    def __init__(self, typ, pth, truncate_end, truncate_st, n_classes):
        self.files = [f for f in os.listdir(pth)]
        self.files = sorted(self.files, key=lambda x: int(x.split('round')[1].split('.csv')[0]))
        if typ == 'accuracy' and n_classes == 10:
            indx = 10
        if typ == 'accuracy' and n_classes == 100:
            indx = 100
        # classification error
        if truncate_end == None:
            self.metric = [pd.read_csv(pth + f).iloc[indx][1] for f in self.files][:40]
        else:
            if not truncate_st: truncate_st = 0
            self.metric = [pd.read_csv(pth + f).iloc[indx][1] for f in self.files][truncate_st:truncate_end]

def PlotGraph(figure_nam,data_locs,labels,truncate_end=None,truncate_st=None, x_l=None,y_l=None,
              n_classes=10,isample=100, mini_window=False):
    metric_list = [getMetric('accuracy',pth,truncate_end, truncate_st, n_classes).metric for pth in data_locs]
    rounds = range(len(metric_list[0]))
    if mini_window:
        mini_window_metric = [getMetric('accuracy',pth, 15, truncate_st, n_classes).metric for pth in data_locs]
        rounds_mini_window = range(len(mini_window_metric[0]))
    colors = ['#9B59B6','#76D7C4','#F5B041','#E74C3C','#2C3E50', '#3498DB']
    model_names = {'b_names':{'BBB':['BAL-VR','BAL-E','BRS'],'Jeffrey':['BJAL-VR', 'BJAL-E']}}
    # create specific colors and markers
    markers, line_style = [], []
    for l in labels:
        if l in model_names['b_names']['BBB']:
            markers.append('^')
            line_style.append('--')
        elif l in model_names['b_names']['Jeffrey']:
            markers.append('X')
            line_style.append('--')
        else:
            markers.append('o')
            line_style.append('-')

    font = {'family': 'Serif',
            'weight': 'normal',
            'size': 14}
    rc('font', **font)
    if not figure_nam:
        figure_nam = 'my_fig.png'
    fig = plt.figure(num=1, figsize=(14, 7))
    ax = plt.axes()
    if mini_window:
        axins = ax.inset_axes([0.3, 0.1, 0.45, 0.4])
    for i,m in enumerate(metric_list):
        ax.plot(rounds, m, marker=markers[i], linestyle=line_style[i], color=colors[i], label=labels[i])
        if mini_window:
            axins.plot(rounds_mini_window, mini_window_metric[i], marker=markers[i], linestyle=line_style[i], color=colors[i], label=labels[i])
    if x_l:
        ax.set_xlim(x_l)
    if y_l:
        ax.set_ylim(y_l)

    ax.set_ylabel('Classification Accuracy', fontsize=18)
    ax.set_xlabel(r'# Samples ($\times ' + str(isample) + '$)', fontsize=18)
    ax.legend(loc=4, ncol=1, frameon=True)
    if mini_window: ax.indicate_inset_zoom(axins)
    plt.show()

if __name__ == '__main__':
    # with 50 samples at start 2020-05-25 18:43:21 
    pth_1 = '../results/McDropout/lenet5_fmnist_isample50_e50_r81_ac1_optim-Adam_top-k100_b32_rtAfter100_variation-ratio/'
    pth_2 = '../results/McDropout/lenet5_fmnist_isample50_e50_r81_ac1_optim-Adam_top-k100_b32_rtAfter100_entropy/'
    pth_3 = '../results/lenet5_fmnist_isample50_e50_r81_ac1_Klreg-standard2_b100_topK-100_mcmc1_netType-jeffreyoptim-Adam_variation-ratio_rtAfter100/'
    pth_4 = '../results/lenet5_fmnist_isample50_e50_r81_ac1_Klreg-standard2_b100_topK-100_mcmc1_netType-jeffreyoptim-Adam_entropy_rtAfter100/'
    locs = [pth_1, pth_2, pth_3, pth_4]
    labels = ['AL-VR','AL-E','BJAL-VR','BJAL-E']
    f_name = 'fig_temp'
    PlotGraph(f_name,locs,labels, truncate_end=80, isample=10)