from random import shuffle, seed

import matplotlib.pyplot as plt

from coverexp.topic_structure.dim_reduction import reduceDim
from doc_topic_coh.dataset.topic_splits import iter0DevTestSplit
from pytopia.resource.tools import tfIdfMatrix

def plotClusters(ltopics, label, ntopics=None, rseed=None, threshold=0.1,
                 reduceParams={'method':'pca'}):
    '''
    Plot dim-reduced clusters of tfidf vectors of top topic-related documents.
    :param ltopics: list of (topic, label)
    :param label: plot for topics with these labels
    :param ntopics: if not None, plot only for this many topics (from list start)
    :param seed: if not None, shuffle list
    :param reduceParams: dimReduce parameters
    :return:
    '''
    ltopics = [ lt for lt in ltopics if lt[1] == label ]
    if rseed: seed(rseed); shuffle(ltopics)
    if ntopics: ltopics = ltopics[:ntopics]
    N = len(ltopics)
    fig, axes = plt.subplots(1, N)
    h = 4.0
    fig.set_size_inches(h*N+1, h)
    plt.suptitle(label)
    for i, lt in enumerate(ltopics):
        t, l = lt
        m = tfIdfMatrix(t)
        drm = reduceDim(m, **reduceParams)
        ax = axes[i] if N > 1 else axes
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.scatter(drm[:, 0], drm[:, 1], s=0.05)
        topicStr = '%s.%s' % t
        ax.set_title(topicStr)
    fname = 'tfidfClusters_%s_Ntopics[%s]_Rseed[%s]_dimReduce[%s]_threshold[%s].pdf' % \
                (label, str(ntopics), str(rseed), str(reduceParams), str(threshold))
    fig.subplots_adjust()
    plt.savefig(fname)

def plot1():
    # theme, theme_noise, theme_mix, theme_mix_noise, noise
    dev, _ = iter0DevTestSplit()
    for l in ['theme', 'theme_noise', 'theme_mix', 'theme_mix_noise', 'noise']:
        #plotClusters(dev, l, 8, 123, threshold=100)
        # plotClusters(dev, l, 8, 123, threshold=100,
        #              reduceParams={'method':'pca-kernel','kernel':'chi-squared'})
        plotClusters(dev, l, 8, 123, threshold=100, reduceParams={'method':'nmf'})

if __name__ == '__main__':
    plot1()