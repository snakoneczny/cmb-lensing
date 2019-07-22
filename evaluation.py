from math import log

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import norm


def relative_error(y_true, y_pred):
    e = (y_pred - y_true) / y_true
    return e.mean(), e.std()


def evaluate(exp_output, y_col='M500c', mass_max=None):
    mean = exp_output[y_col].mean()
    mse = mean_squared_error(exp_output[y_col], [mean] * exp_output.shape[0])
    print('MSE when estimating mean {} value: {:.4f}'.format(y_col, mse))

    metrics = [('MSE', mean_squared_error), ('MAE', mean_absolute_error)]
    for metric_name, metric_func in metrics:
        print('{}: {:.4f}'.format(metric_name, metric_func(exp_output[y_col], exp_output['m_pred'])))

    mu, sigma = relative_error(exp_output[y_col], exp_output['m_pred'])
    print('Relative error mean: {:.4f}, sigma: {:.4f}'.format(mu, sigma))

    residuals = exp_output['m_pred'].apply(log) - exp_output[y_col].apply(log)
    mu, sigma = norm.fit(residuals)
    print('Log residuals mean: {:.4f}, sigma: {:.4f}'.format(mu, sigma))

    exp_output.plot.scatter(x=y_col, y='m_pred')

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
