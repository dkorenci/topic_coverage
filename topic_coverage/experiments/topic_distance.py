from topic_coverage.topicmatch.data_iter0 import loadDataset
from topic_coverage.modelbuild.modelbuild_iter1 import modelsContext
from pytopia.measure.topic_distance import *

from stat_utils.utils import Stats

from matplotlib import pyplot as plt
from numpy import arange

def plotClassProbForDistIntervals(topicPairs, measure, intervals):
    # create map interval -> label stats (map labelVal->count)
    labelDist = {}
    labels = sorted(set(l for _, _, l in topicPairs))
    for i in intervals:
        labelDist[i] = {}
        for l in labels: labelDist[i][l] = 0.0
    dists = set()
    for t1, t2, l in topicPairs:
        dist = measure(t1.vector, t2.vector)
        dists.add(dist)
        for i in intervals:
            low, high = i
            if dist >= low and dist < high:
                labelDist[i][l] += 1.0
    dists = list(dists)
    print 'LABELS', labels
    print 'DISTANCE DISTRIBUTION', Stats(dists)
    # plot
    cmap = plt.get_cmap('terrain')
    w = 1.0 / len(intervals); lbarw = w / len(labels) * 0.9
    xcoords = arange(0, 1.0, w)
    fig, axes = plt.subplots(1, 1)
    axes.set_xticklabels(['%.2f-%.2f' % iv for iv in intervals])
    axes.set_xticks(xcoords+0.5*lbarw)
    barsets = []
    for tick in axes.get_xticklabels(): tick.set_fontsize(7)
    for i, lab in enumerate(labels):
        values = []
        for iv in intervals:
            labelsPerInterval = sum(labelDist[iv][l] for l in labels)
            labelIntRatio = labelDist[iv][lab] / labelsPerInterval if labelsPerInterval > 0 else 0
            values.append(labelIntRatio)
        bars = axes.bar(xcoords+i*lbarw, values, lbarw, align='center', color=cmap(1.0*i/len(labels)))
        barsets.append(bars)
    axes.yaxis.grid(True)
    axes.legend(barsets, labels)
    plt.xlabel('intervals of %s' % measure.__name__)
    plt.ylabel('same pairs (1) vs. distinct pairs (0) distribution')
    plt.title('sameness of pairs sampled from intervals of %s' % measure.__name__)
    plt.show()

def createIntervals(start, stop, numInt):
    '''
    Return a list of (a,b) equidistant sub-intervals of (start, stop)
    :return:
    '''
    w = (stop-start) / numInt
    xcoords = arange(start, stop, w)
    return [ (s, s+w) for s in xcoords ]

def plotIntervalClassDistCosine():
    from topic_coverage.topicmatch.data_iter0 import folder, fnames
    with modelsContext():
        pairs = loadDataset(folder, fnames)
        plotClassProbForDistIntervals(pairs, cosine, createIntervals(0.0, 1.0, 10))

def plotIntervalClassDistKL():
    pairs = loadDataset()
    #kl = klDivZero
    kl = klDivSymm
    plotClassProbForDistIntervals(pairs, kl, createIntervals(0.0, 16, 20))

def classDistribution():
    pairs = loadDataset()
    N = len(pairs)
    cnt = {0:0.0, 1:0.0}
    for _, _, l in pairs: cnt[l] += 1
    print 'N: %d ; 0: %3d %.3f ; 1: %3d %.3f' % (N, cnt[0], cnt[0]/N, cnt[1], cnt[1]/N)

def analyzeLabelDistVsDistance(corpus='uspol', labeler=None, dist=cosine):
    from topic_coverage.topicmatch.supervised_data import getFolderAndFiles, getLabelingContext
    folder, files = getFolderAndFiles(corpus)
    with getLabelingContext(corpus):
        if labeler: files = files[labeler] # single labeler files
        else: files = [f for fset in files.itervalues() for f in fset] # all labeler files
        pairs = loadDataset(folder, files)
        plotClassProbForDistIntervals(pairs, dist, createIntervals(0.0, 3, 30))

if __name__ == '__main__':
    analyzeLabelDistVsDistance(corpus='pheno', dist=jensenShannon)
