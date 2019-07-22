from math import log

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import norm


def relative_error(y_true, y_pred):
    e = (y_pred - y_true) / y_true
    return e.mean(), e.std()


def evaluate(exp_output, y_col='M500c', mass_max=None):
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
