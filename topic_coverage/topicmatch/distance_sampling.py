import cPickle
import random
from os import path

from pyutils.file_utils.location import FolderLocation as loc
from topic_coverage.settings import resource_folder
from topic_coverage.topicmatch.topicplots import topicPairs
from topic_coverage.topicmatch.tools import topicPid
from gtar_context.semantic_topics.construct_model import MODEL_ID as USPOL_REF_MODEL

datafolder = loc(path.join(resource_folder, 'topicmatch'))

testPairs = 'topicDist_[topics=test]_[dist=cosine]_[seed=356]'
iter0UspolPairs = 'topicDist_[topics=uspolTopicsIter0]_[dist=cosine]_[seed=6356]'

def createDistances(topics, dist, rndseed, topicsId, verbose=False, sampleSize=None):
    '''
    Create and save a list of (topic1, topic2, dist) triples.
    :param topics: list of Topic objects
    :param dist: distance function on pairs of Topic objects
    :param topicsId: id of the topic set, for file naming
    :param verbose: if true, print progress
    :param sampleSize: if integer, downsample topic pairs to this size
    :return:
    '''
    if verbose: print 'num topics: %d' % len(topics)
    pairs = topicPairs(topics, rseed=rndseed, sampleSize=sampleSize)
    if verbose: print 'num pairs: %d' % len(pairs)
    pairData = []; cnt = 0
    for t1, t2 in pairs:
        t1id, t2id = topicPid(t1), topicPid(t2)
        if t1id != t2id: pairData.append((t1id, t2id, dist(t1.vector, t2.vector)))
        cnt += 1
        if verbose and cnt % 10000 == 0: print '%d pairs processed' % cnt
    if verbose: print '%d pairs processed' % cnt
    #todo: sampleSize should also be used for fname, and rndseed only is sampleSize != None
    fname = distancesFname(dist, rndseed, topicsId)
    cPickle.dump(pairData, open(datafolder(fname), 'wb'))
    print 'CREATED DISTANCES, FILE: ', fname
    return fname

def distancesFname(dist, rndseed, topicsId):
    return 'topicDist_[topics=%s]_[dist=%s]_[seed=%s].pickle' % (topicsId, dist.__name__, rndseed)

def loadDistances(fname):
    return cPickle.load(open(datafolder(fname), 'rb'))

def intervals(low, high, N, incLast=True, flat=False, step=None):
    import numpy as np
    N = int(N)
    if step is None: step = (float(high)-float(low))/N
    ivals = []
    for i in np.arange(N):
        low = step*i; high = step*(i+1)
        if i == N-1 and incLast: high += 1e-8
        ivals.append((low, high))
    if not flat: return ivals
    else:
        iv = []
        for l, h in ivals: iv.extend([l, h])
        return sorted(set(iv))

def distancesPerInterval(pairsFile, intervals):
    ''' Print statistics about grouping topic pairs into distance intervals '''
    from topic_coverage.topicmatch.distance_sampling import loadDistances, \
        samplePairsByDistInterval
    # id of the generated labeling sample
    tpairs = loadDistances(pairsFile)
    print 'number of all pairs: %d' % len(tpairs)
    pairsByInt = groupPairsByInterval(tpairs, intervals)
    N = float(len(tpairs))
    for int in intervals:
        sz = len(pairsByInt[int])
        print 'perc %4g, size %6d , interval %s' % (sz/N, sz, int)

def groupPairsByInterval(tpairs, intervals):
    '''
    Divide topic pairs into intervals by distance.
    :return: map interval->pairs
    '''
    pairsByDist = { ival : [] for ival in intervals }
    for i, tp in enumerate(tpairs):
        t1, t2, d = tp
        for ival in intervals:
            if ival[0] <= d < ival[1]:
                pairsByDist[ival].append((i, t1, t2, d))
    return pairsByDist

def familyFromid(mid, minor=False):
    f = None
    if 'type[lda-asym]' in mid: f = 'alda'
    elif 'type[lda]' in mid: f = 'lda'
    elif 'type[pyp]' in mid: f = 'pyp'
    elif 'SklearnNmfTmAdapter' in mid: f = 'nmf'
    else: f = mid
    if minor:
        numtop = [50, 100, 200]
        for t in numtop:
            ts = 'T[%d]'%t
            if ts in mid: f += ('%d'%t)
    return f

def modelFamiliesPerSample(pairsFile, intervals, sizes, rndseed=9081):
    '''
    Print statistics about model families distribution in the sample
    obtained by samplePairsByDistInterval()
    '''
    from topic_coverage.topicmatch.distance_sampling import loadDistances, \
        samplePairsByDistInterval
    # id of the generated labeling sample
    tpairs = loadDistances(pairsFile)
    pairsByInt = samplePairsByDistInterval(tpairs, intervals, sizes, rndseed)
    pairs =  [(pairData[1], pairData[2]) for int in intervals for pairData in pairsByInt[int]]
    np = float(len(pairs))
    print 'number of all pairs: %d' % np
    # init counts
    family, mfamily = {}, {}
    for tid1, tid2 in pairs:
        mid1, mid2 = tid1[0], tid2[0]
        f1, mf1 = familyFromid(mid1), familyFromid(mid1, True)
        f2, mf2 = familyFromid(mid2), familyFromid(mid2, True)
        family[f1] = 0; family[f2] = 0
        mfamily[mf1] = 0; mfamily[mf2] = 0
    # count
    for tid1, tid2 in pairs:
        mid1, mid2 = tid1[0], tid2[0]
        f1, mf1 = familyFromid(mid1), familyFromid(mid1, True)
        f2, mf2 = familyFromid(mid2), familyFromid(mid2, True)
        family[f1] += 1; mfamily[mf1] += 1
        if f2 != f1: family[f2] += 1
        if mf2 != mf1: mfamily[mf2] += 1
    families = sorted(set(family.keys()))
    mfamilies = sorted(set(mfamily.keys()))
    for f in families:
        print 'family %s , count %4d, perc %4g' % (f, family[f], family[f]/np)
    print
    for mf in mfamilies:
        print 'mfamily %s , count %4d, perc %4g' % (mf, mfamily[mf], mfamily[mf]/np)

def samplePairsByDistInterval(tpairs, intervals, sizes, rndseed=9081, verbose=False):
    '''
    Sample topic pairs by dividing them into intervals by distance
     and sampling the given number of pairs from each interval.
    :param tpairs: list of (topic1, topic2, distance)
    :param intervals: list of (dist1, dist2)
    :param sizes: sample size per interval, number or a list (number per interval)
    :return: map interval : sampledTopicPairsList
    '''
    if isinstance(sizes, (int, float)): sizes = [sizes]*len(intervals)
    else: assert len(sizes) == len(intervals)
    pairsByDist = { ival : [] for ival in intervals }
    for i, tp in enumerate(tpairs):
        t1, t2, d = tp
        for ival in intervals:
            if ival[0] <= d < ival[1]:
                pairsByDist[ival].append((i, t1, t2, d))
    if verbose:
        print 'num pairs: %d' % len(tpairs)
        print 'sizes by distance intervals'
        for ival in sorted(pairsByDist.keys()):
            print ival, len(pairsByDist[ival])
    random.seed(rndseed)
    def mksample(items, size):
        if len(items) <= size: return [i for i in items]
        else: return random.sample(items, size)
    return { ival: mksample(pairsByDist[ival], sizes[i])
             for i, ival in enumerate(pairsByDist) }

if __name__ == '__main__':
    pass