import json
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import math
import json
from scipy.stats import ttest_ind
from utils import get_stat_func, CompressPDF

font = {'size': 60}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.constrained_layout.use'] = True

colors = [[0, 0.447, 0.7410], [0.85, 0.325, 0.098],  [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
          [0.494, 0.1844, 0.556],[0.3010, 0.745, 0.933], [137/255,145/255,145/255],
          [0.466, 0.674, 0.8], [0.929, 0.04, 0.125],
          [0.3010, 0.245, 0.33], [0.635, 0.078, 0.184], [0.35, 0.78, 0.504]]

RESULTS_PATH = '/home/ahmed/Documents/eta-optimality/eta_otpimality/results_8/'
SAVE_PATH = '/home/ahmed/Documents/eta-optimality/eta_otpimality/plots_8/'

LINE = 'mean'
ERR = 'std'
DPI = 30
N_SEEDS = None
N_EPOCHS = None
LINEWIDTH = 10
MARKERSIZE = 3
ALPHA = 0.3
ALPHA_TEST = 0.05
MARKERS = ['o', 'v', 's', 'P', 'D', 'X', "*", 'v', 's', 'p', 'P', '1']
FREQ = 1
NB_BUCKETS = 5
NB_EPS_PER_EPOCH = 16
NB_VALID_GOALS = 35
LAST_EP = 160
LIM = NB_EPS_PER_EPOCH * LAST_EP / 1000 + 0.03
line, err_min, err_plus = get_stat_func(line=LINE, err=ERR)
COMPRESSOR = CompressPDF(4)
# 0: '/default',
# 1: '/prepress',
# 2: '/printer',
# 3: '/ebook',
# 4: '/screen'


def setup_figure(xlabel=None, ylabel=None, xlim=None, ylim=None):
    fig = plt.figure(figsize=(22, 15), frameon=False)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(6)
    ax.spines['right'].set_linewidth(6)
    ax.spines['bottom'].set_linewidth(6)
    ax.spines['left'].set_linewidth(6)
    ax.tick_params(width=10, direction='in', length=20, labelsize='55')
    artists = ()
    if xlabel:
        xlab = plt.xlabel(xlabel)
        artists += (xlab,)
    if ylabel:
        ylab = plt.ylabel(ylabel)
        artists += (ylab,)
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    return artists, ax

def setup_n_figs(n, xlabels=None, ylabels=None, xlims=None, ylims=None):
    fig, axs = plt.subplots(n, 1, figsize=(22, 15), frameon=False)
    axs = axs.ravel()
    artists = ()
    for i_ax, ax in enumerate(axs):
        ax.spines['top'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.tick_params(width=7, direction='in', length=15, labelsize='55', zorder=10)
        if xlabels[i_ax]:
            xlab = ax.set_xlabel(xlabels[i_ax])
            artists += (xlab,)
        if ylabels[i_ax]:
            ylab = ax.set_ylabel(ylabels[i_ax])
            artists += (ylab,)
        if ylims[i_ax]:
            ax.set_ylim(ylims[i_ax])
        if xlims[i_ax]:
            ax.set_xlim(xlims[i_ax])
    return artists, axs

def save_fig(path, artists):
    plt.savefig(os.path.join(path), bbox_extra_artists=artists, bbox_inches='tight', dpi=DPI)
    plt.close('all')
    # compress PDF
    try:
        COMPRESSOR.compress(path, path[:-4] + '_compressed.pdf')
        os.remove(path)
    except:
        pass


def check_length_and_seeds(experiment_path):
    conditions = os.listdir(experiment_path)
    # check max_length and nb seeds
    max_len = 0
    max_seeds = 0
    min_len = 1e6
    min_seeds = 1e6

    for cond in conditions:
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        if len(list_runs) > max_seeds:
            max_seeds = len(list_runs)
        if len(list_runs) < min_seeds:
            min_seeds = len(list_runs)
        for run in list_runs:
            try:
                run_path = cond_path + run + '/'
                data_run = pd.read_csv(run_path + 'progress.csv')
                nb_epochs = len(data_run)
                if nb_epochs > max_len:
                    max_len = nb_epochs
                if nb_epochs < min_len:
                    min_len = nb_epochs
            except:
                pass
    return max_len, max_seeds, min_len, min_seeds

def plot_sr_av(max_len, experiment_path, algos, plot_name):
    if plot_name == 'HalfCheetah-v2':
        artists, ax = setup_figure(  # xlabel='Episodes (x$10^3$)',
            xlabel='Million steps',
            ylabel='Average Reward',
            xlim=[-0.02, LIM],
            ylim=[-0.02, 12000.03])
    elif plot_name == 'Hopper-v2':
        artists, ax = setup_figure(  # xlabel='Episodes (x$10^3$)',
            xlabel='Million steps',
            ylabel='Average Reward',
            xlim=[-0.02, LIM],
            ylim=[-0.02, 4000.03])
    elif plot_name == 'Walker2d-v2':
        artists, ax = setup_figure(  # xlabel='Episodes (x$10^3$)',
            xlabel='Million steps',
            ylabel='Average Reward',
            xlim=[-0.02, LIM],
            ylim=[-0.02, 5000.03])

    for k, folder in enumerate(algos):
        condition_path = experiment_path + folder + '/'
        list_runs = sorted(os.listdir(condition_path))
        global_sr = np.zeros([len(list_runs), max_len])
        global_sr.fill(np.nan)
        sr_data = np.zeros([len(list_runs), max_len])
        sr_data.fill(np.nan)
        x_eps = np.arange(0, (LAST_EP + 1) * NB_EPS_PER_EPOCH, NB_EPS_PER_EPOCH * FREQ) / 1000
        x = np.arange(0, LAST_EP + 1, FREQ)
        for i_run, run in enumerate(list_runs):
            run_path = condition_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')
            T = len(data_run['avg_reward'][:LAST_EP + 1])
            SR = data_run['avg_reward'][:LAST_EP + 1]
            sr_data[i_run, :T] = SR.copy()

        sr_per_cond_stats = np.zeros([max_len, 3])
        sr_per_cond_stats[:, 0] = line(sr_data)
        sr_per_cond_stats[:, 1] = err_min(sr_data)
        sr_per_cond_stats[:, 2] = err_plus(sr_data)
        # av = line(global_sr)
        plt.plot(x_eps, sr_per_cond_stats[x, 0], color=colors[k], marker=MARKERS[k], markersize=MARKERSIZE, linewidth=LINEWIDTH)
        plt.fill_between(x_eps, sr_per_cond_stats[x, 1], sr_per_cond_stats[x, 2], color=colors[k], alpha=ALPHA)
    plt.title(plot_name)
    leg = plt.legend(algos,
                     loc='upper center',
                     bbox_to_anchor=(0.5, 1.45),
                     ncol=4,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 45, 'weight': 'bold'},
                     markerscale=1)
    artists += (leg,)
    if plot_name == 'HalfCheetah-v2':
        ax.set_yticks([2000, 4000, 6000, 8000, 10000, 12000])
    elif plot_name == 'Hopper-v2':
        ax.set_yticks([1000, 2000, 3000, 4000])
    elif plot_name == 'Walker2d-v2':
        ax.set_yticks([1000, 2000, 3000, 4000, 5000])
    plt.grid()
    save_fig(path=SAVE_PATH + PLOT + '_sr.pdf', artists=artists)

if __name__ == '__main__':
    PLOTS = ['HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
    ALGOS = ['SAC', 'GSAC']
    for PLOT in PLOTS:
        print('\n\tPlotting', PLOT)
        experiment_path = RESULTS_PATH + PLOT + '/'

        max_len, max_seeds, min_len, min_seeds = check_length_and_seeds(experiment_path=experiment_path)

        plot_sr_av(max_len, experiment_path, ALGOS, PLOT)



