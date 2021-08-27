from matplotlib import pyplot as plt
from random import sample, seed

def topicPairs(topics, sampleSize=None, rseed=88354):
    '''
    Generate (a sample of) all topic pairs from models
    Return a list of [((modelIndex1, topicId1), (modelIndex2, topicId2))]
    :param models: list of pytopia models
    :param sampleSize: if None, take all pairs
    :return:
    '''
    topicPairs = [(t1, t2) for ti, t1 in enumerate(topics) for tj, t2 in enumerate(topics) if (tj > ti)]
    if sampleSize is not None and sampleSize < len(topicPairs):
        seed(rseed)
        topicPairs = sample(topicPairs, sampleSize)
    return topicPairs

def valueDist(measures, topics, sampleSize=None, cumulative=False):
    '''
    Create boxplot and histograms for topic distance measures on (subset of) topic pairs.
    :param measures: list of distance measure functions
    :param models: list of pytopia models
    :param sampleSize: if None, take all the pairs
    :return:
    '''
    M = len(measures)
    fig, axes = plt.subplots(M, 2)
    pairs = topicPairs(topics, sampleSize)
    for i, ms in enumerate(measures):
        vals = []
        for t1, t2 in pairs:
            vals.append(ms(t1.vector, t2.vector))
        histParams = {'bins': 100, 'normed': True, 'cumulative': cumulative}
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
        vals.sort(); cv = 0; V = float(len(vals))
        for i, v in enumerate(vals):
            cv += v
            print '%.4f, %.4f' % (i/V*100, v)
    plt.show()
