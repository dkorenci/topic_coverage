from matplotlib import pyplot as plt
import numpy as np

def basicValueDist(values, title=None, labels=None, cumulative=False,
                   bins=100, boxplotVals=False, save=False, boxplt=True,
                   xlabel=None, ylabel=None):
    '''
    Create boxplots and histograms for lists of values.
    :param values: list of values, or a list of lists of values
    :param lables: list of plot labels, one per value dist
    :param sampleSize: if None, take all the pairs
    :return:
    '''
    if not isinstance(values[0], list):
        # if not list of lists, embed in a list to handle everything in same format
        values = [values]
    M = len(values)
    if boxplt: fig, axes = plt.subplots(M, 2, figsize=(15, 8))
    else: fig, axes = plt.subplots(M, 1, figsize=(15, 8))
    boxplotxcoord = 1.0
    for i, vals in enumerate(values):
        if boxplt:
            if M > 1: bax, hax = axes[i, 0], axes[i, 1]
            else: bax, hax = axes[0], axes[1]
        else:
            bax = None
            if M > 1: hax = axes[i]
            else: hax = axes
        # set tick font for axes
        for ax in [bax, hax]:
            if ax is not None:
                for tick in ax.get_xticklabels(): tick.set_fontsize(15)
                for tick in ax.get_yticklabels(): tick.set_fontsize(15)
        # boxplot
        if boxplt:
            outlierstyle={'marker':'.', 'markerfacecolor':'blue', 'markersize':1}
            bax.boxplot(vals, flierprops=outlierstyle)
            if boxplotVals:
                bax.scatter([boxplotxcoord] * len(vals), vals, alpha=0.4, s=0.1)
        # histogram
        probabilityWeights = np.zeros_like(vals) + 1./len(vals)
        histParams = {'bins': bins, 'weights':probabilityWeights, 'cumulative': cumulative}
        hax.hist(vals, **histParams)
        # label plots
        #if labels:
        #    for j in range(2): hax.set_title(labels[i])
    #if title: fig.suptitle(title)
    if xlabel: plt.xlabel(xlabel, fontsize=20)
    if ylabel: plt.ylabel(ylabel, fontsize=20)
    if save: plt.savefig(title+'.pdf', box_inches='tight')
    else: plt.show()
