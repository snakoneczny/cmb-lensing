from itertools import chain
from math import log

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from utils import safe_indexing


def relative_error(y_true, y_pred):
    e = (y_pred - y_true) / y_true
    return e.mean(), e.std()


def evaluate(exp_output, y_col='M500c', mass_max=None):
    mean = exp_output[y_col].mean()
    mse = mean_squared_error(exp_output[y_col], [mean] * exp_output.shape[0])
    print('MSE when estimating mean {} value: {:.4f}'.format(y_col, mse))

    metrics = [('MSE', mean_squared_error), ('MAE', mean_absolute_error), ('R2', r2_score)]
    for metric_name, metric_func in metrics:
        print('{}: {:.4f}'.format(metric_name, metric_func(exp_output[y_col], exp_output['m_pred'])))

    mu, sigma = relative_error(exp_output[y_col], exp_output['m_pred'])
    print('Relative error mean: {:.4f}, sigma: {:.4f}'.format(mu, sigma))

    residuals = exp_output['m_pred'].apply(log) - exp_output[y_col].apply(log)
    mu, sigma = norm.fit(residuals)
    print('Log residuals mean: {:.4f}, sigma: {:.4f}'.format(mu, sigma))

    exp_output.plot.scatter(x=y_col, y='m_pred')
    plt.plot(range(mass_max + 1))

    plt.figure()
    p = sns.kdeplot(exp_output[y_col], exp_output['m_pred'], shade=True)
    if mass_max:
        plt.plot(range(mass_max + 1))
        p.set(xlim=(0, mass_max), ylim=(0, mass_max))

    # label, color=color_palette[i], 'linestyle': get_line_style(i)
    plt.figure()
    sns.distplot(residuals, kde=False, rug=False, norm_hist=True, fit=norm,
                 hist_kws={'alpha': 1.0, 'histtype': 'step', 'linewidth': 1.5})
    plt.xlabel('log(M_pred) - log({y_col})'.format(y_col=y_col))
    plt.ylabel('normalized counts per bin')
    # plt.legend()
    plt.tight_layout()

    plot_error_in_bins(exp_output, y_col=y_col)


def plot_error_in_bins(exp_output, y_col='M500c'):
    # Get  and assign bins
    _, bin_edges = np.histogram(exp_output['M500c'], bins=40)
    exp_output.loc[:, 'binned'] = pd.cut(exp_output['m_pred'], bin_edges)

    # Calculate residuals
    exp_output.loc[:, 'rel_residual'] = (exp_output['m_pred'] - exp_output['M500c']) / exp_output['M500c']
    exp_output.loc[:, 'residual_sqr'] = (exp_output['m_pred'] - exp_output['M500c']) ** 2

    # Calculate values in bins
    grouped = exp_output.groupby(by='binned')
    rel_error = grouped['rel_residual'].mean().values
    mse = grouped['residual_sqr'].mean().values
    size = grouped.size().values

    # Plot values in bins
    to_plot = [(mse, 'mean squared error'), (rel_error, 'rel. error'), (size, 'number of clusters')]
    for x, plot_title in to_plot:
        plt.figure()
        ax = sns.lineplot(bin_edges[:-1], x, drawstyle='steps-pre')  # color=color_palette[i]
        #     ax.lines[i].set_linestyle(get_line_style(i))
        plt.xlabel('M500c')
        plt.ylabel(plot_title)
    #     plt.legend()


def train_test_many_split(*arrays, side_test_size=0.05, random_test_size=0.1):
    """
    :param arrays: arrays of any format, the first one should be array or Series
        All arrays are divided based on the values in the first one
    :param side_test_size: float (0, 1)
    :param random_test_size: float (0, 1)
    :return: list
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError('At least one array required as input')

    # Make indexable
    arrays = [a for a in arrays]

    # Get top index
    k = int(side_test_size * arrays[0].shape[0])
    split_ind = arrays[0].shape[0] - k
    ind_part = np.argpartition(arrays[0], split_ind)
    ind_test_top = ind_part[split_ind:]
    ind_top_low = ind_part[:split_ind]

    # Get low index
    split_ind = k
    ind_part = np.argpartition(arrays[0], split_ind)
    ind_low_top = ind_part[split_ind:]
    ind_test_low = ind_part[:split_ind]

    ind_middle = np.intersect1d(ind_top_low, ind_low_top, assume_unique=True)
    ind_train, ind_test_random = train_test_split(ind_middle, test_size=random_test_size, random_state=8725)

    return list(chain.from_iterable((safe_indexing(a, ind_train), safe_indexing(a, ind_test_low),
                                     safe_indexing(a, ind_test_top), safe_indexing(a, ind_test_random)) for a in
                                    arrays))


def train_test_top_split(*arrays, test_size=0.05):
    """
    :param arrays: arrays of any format, the first one should be array or Series
        All arrays are divided based on the values in the first one
    :param test_size: float (0, 1)
    :return: list
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError('At least one array required as input')

    # Make indexable
    arrays = [a for a in arrays]

    # Get top
    k = int(test_size * arrays[0].shape[0])
    split_ind = arrays[0].shape[0] - k
    ind_part = np.argpartition(arrays[0], split_ind)
    ind_top = ind_part[split_ind:]
    ind_low = ind_part[:split_ind]

    return list(chain.from_iterable((safe_indexing(a, ind_low), safe_indexing(a, ind_top)) for a in arrays))
