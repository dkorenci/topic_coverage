# -*- coding: utf-8 -*-

from topic_coverage.topicmatch.data_iter0 import loadDataset
from topic_coverage.modelbuild.modelbuild_iter1 import modelsContext
from pytopia.measure.topic_distance import *

from pyutils.stat_utils.utils import Stats

from matplotlib import pyplot as plt
from numpy import arange

def valueDist(topicPairs, measures, savefile=None):
    '''
    Create boxplot and histograms for topic distance measures on (subset of) topic pairs.
    :param measures: list of distance measure functions
    :param models: list of pytopia models
    :param sampleSize: if None, take all the pairs
    :return:
    '''
    M = len(measures)
    fig, axes = plt.subplots(M,2)
    for i, ms in enumerate(measures):
        vals = [ms(t1.vector, t2.vector) for t1, t2, _ in topicPairs]
        histParams = {'bins':100, 'density':False}
        print ms.__name__
        print Stats(vals)
        if M > 1:
            axes[i, 0].boxplot(vals)
            axes[i, 1].hist(vals, **histParams)
            for j in range(2):
                axes[i, j].set_title(ms.__name__)
        else:
            axes[0].boxplot(vals)
            axes[1].hist(vals, **histParams)
            for j in range(2):
                axes[j].set_title(ms.__name__)
    if savefile: plt.savefig(savefile+'.pdf')
    else: plt.show()

def plotClassDistribution(topicPairs, measure, intervals, resolve=False):
    # create map interval -> label stats (map labelVal->count)
    from topic_coverage.topicmatch.data_iter0 import resolveTopic
    if resolve:
        topicPairs = [(resolveTopic(t1), resolveTopic(t2), l) for t1, t2, l in topicPairs]
    labelDist = {}
    labels = sorted(set(l for _, _, l in topicPairs))
    for i in intervals:
        labelDist[i] = {}
        for l in labels: labelDist[i][l] = 0.0
    for t1, t2, l in topicPairs:
        dist = measure(t1.vector, t2.vector)
        for i in intervals:
            low, high = i
            if dist >= low and dist < high:
                labelDist[i][l] += 1.0
    print labels
    # plot
    cmap = plt.get_cmap('terrain')
    w = 1.0 / len(intervals); lbarw = w / len(labels) * 0.9
    xcoords = arange(0, 1.0, w)
    fig, axes = plt.subplots(1, 1)
    axes.set_xticklabels(['%.2f-%.2f' % iv for iv in intervals])
    axes.set_xticks(xcoords+0.5*lbarw)
    barsets = []
    for tick in axes.get_xticklabels(): tick.set_fontsize(15)
    for tick in axes.get_yticklabels(): tick.set_fontsize(15)
    for i, lab in enumerate(labels):
        values = []
        for iv in intervals:
            lsum = sum(labelDist[iv][l] for l in labels)
            v = labelDist[iv][lab] / lsum if lsum != 0 else 0.0
            values.append(v)
        bars = axes.bar(xcoords+i*lbarw, values, lbarw, align='center', color=cmap(1.0*i/len(labels)))
        barsets.append(bars)
    axes.yaxis.grid(True)
    labmap = {1.0: u'poklapanje', 0.5:u'relaksirano poklapanje', 0.0:u'bez poklapanja'}
    print labels
    labels = [labmap[l] for l in labels]
    axes.legend(barsets, labels, prop={'size': 17})
    #plt.xlabel('intervals of %s' % measure.__name__)
    plt.xlabel('intervali udaljenosti', fontsize=20)
    plt.ylabel('vjerojatnost poklapanja parova tema', fontsize=20)
    #plt.title('sameness of pairs sampled from intervals of %s' % measure.__name__)
    plt.show()

def createIntervals(start, stop, numInt):
    '''
    Return a list of (a,b) equidistant sub-intervals of (start, stop)
    :return:
    '''
    w = (stop-start) / numInt
    xcoords = arange(start, stop, w)
    return [ (s, s+w) for s in xcoords ]

def plotDistanceDist(measures):
    pairs = loadDataset()
    valueDist(pairs, measures)

def plotIntervalClassDistCosine():
    from topic_coverage.topicmatch.data_iter0 import folder, fnames
    with modelsContext():
        pairs = loadDataset(folder, fnames)
        plotClassDistribution(pairs, cosine, createIntervals(0.0, 1.0, 10))

def plotIntervalClassDistKL():
    pairs = loadDataset()
    #kl = klDivZero
    kl = klDivSymm
    plotClassDistribution(pairs, kl, createIntervals(0.0, 16, 20))

def classDistribution():
    pairs = loadDataset()
    N = len(pairs)
    cnt = {0:0.0, 1:0.0}
    for _, _, l in pairs: cnt[l] += 1
    print 'N: %d ; 0: %3d %.3f ; 1: %3d %.3f' % (N, cnt[0], cnt[0]/N, cnt[1], cnt[1]/N)

if __name__ == '__main__':
    #plotCosineDist([cosine, jensenShannon, l1, canberra])
    #plotDistanceDist([klDivZero, klDivSymm])
    #plotIntervalClassDist()
    #plotIntervalClassDistKL()
    #classDistribution()
    plotIntervalClassDistCosine()
