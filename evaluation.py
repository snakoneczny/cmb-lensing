import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(exp_output, y_col='M500c', mass_max=None):
    # TODO: print MSE

    exp_output.plot.scatter(x=y_col, y='m_pred')

    plt.figure()
    p = sns.kdeplot(exp_output[y_col], exp_output['m_pred'], shade=True)
    if mass_max:
        plt.plot(range(mass_max + 1))
        p.set(xlim=(0, mass_max), ylim=(0, mass_max))
