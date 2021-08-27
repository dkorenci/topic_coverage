from stat_utils.plots import basicValueDist
from stat_utils.utils import Stats

from topic_coverage.topicmatch.supervised_data import getFolderAndFiles, getLabelingContext
from topic_coverage.topicmatch.data_iter0 import loadDataset

import numpy as np


def aggregateLabelerData(folder, labFiles, agg=None):
    allpairs = None
    pairData = {}
    def pairid(t1, t2): return (t1, t2)
    for labeler, files in labFiles.iteritems():
        pairs = loadDataset(folder, files, resolve=False)
        pairids = {pairid(t1, t2) for t1, t2, _ in pairs}
        if allpairs is None:
            allpairs = pairids
            pairData = { pid:[] for pid in allpairs }
        else: assert pairids == allpairs
        for t1, t2, l in pairs:
            pairData[pairid(t1, t2)].append(float(l))
    if agg == 'avg':
        for pid in allpairs: pairData[pid] = np.average(pairData[pid])
    elif agg == 'expand':
        # expand aggregation {pairid->labelsList} into [(topic,topic,label)],
        # repeating same pairs for each label
        return [ (pid[0], pid[1], l) for pid in allpairs for l in pairData[pid] ]
    return pairData

def basicPlots(corpus='uspol'):
    folder, labFiles = getFolderAndFiles(corpus)
    pairData = aggregateLabelerData(folder, labFiles, 'avg')
    #print pairData
    basicValueDist(pairData.values(), boxplotVals=True, labels=[corpus])
    print Stats(pairData.values())

def labeldistVStopicDist(corpus='uspol', measure='cosine'):
    from topic_coverage.topicmatch.data_analysis_iter0 import plotClassDistribution, createIntervals
    from topic_coverage.topicmatch.supervised_data import getLabelingContext
    from pytopia.measure.topic_distance import cosine, l1norm, hellinger, jensenShannon
    ctx = getLabelingContext(corpus)
    folder, labFiles = getFolderAndFiles(corpus)
    pairs = aggregateLabelerData(folder, labFiles, 'expand')
    if measure == 'cosine':
        topicDist = cosine
        intervals = createIntervals(0.0, 1.0, 10)
    elif measure == 'l1norm':
        topicDist = l1norm
        intervals = createIntervals(0.5, 2.0, 10)
    elif measure == 'hellinger':
        topicDist = hellinger
        intervals = createIntervals(0.0, 1.0, 10)
    elif measure == 'JS':
        topicDist = jensenShannon
        intervals = createIntervals(0.0, 1.8, 10)
    with ctx:
        plotClassDistribution(pairs, topicDist, intervals, resolve=True)

from topic_coverage.resources.pytopia_context import topicCoverageContext
if __name__ == '__main__':
    with topicCoverageContext():
        #basicPlots('pheno')
        #labeldistVStopicDist('pheno', 'l1norm')
        labeldistVStopicDist('uspol', 'JS')
        #labeldistVStopicDist('pheno', 'hellinger')