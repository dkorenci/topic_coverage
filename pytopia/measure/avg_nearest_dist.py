from pytopia.tools.IdComposer import IdComposer
from pytopia.measure.topic_distance import *

import numpy as np

class AverageNearestDistance(IdComposer):
    ''' Asymmetric distance between two pytopia models. '''

    def __init__(self, distance, pairwise=False):
        '''
        :param distance: distance measure between topic vectors
        :param pairwise: if True calculate distances on pairs of topics,
            else work with matrices
        '''
        self.dist = distance
        IdComposer.__init__(self)
        self.pariwise = pairwise

    def __call__(self, m1, m2):
        '''
        Calculate AND m1 <- m2, for each topic in m1 find nearest topic in m2
        and calculate the average of these distances.
        :param m1, m2: pytopia model
        :return: average nearest distance
        '''
        if self.pariwise: return self.__andPairwise(m1, m2)
        else: return self.__andBatch(m1, m2)

    def __andBatch(self, m1, m2):
        '''calculate AND by calling distance function for pair of topic matrices'''
        mat1, mat2 = topicMatrix(m1), topicMatrix(m2)
        distMat = self.dist(mat1, mat2)
        self.closestDist = np.min(distMat, axis=1)
        return np.mean(np.min(distMat, axis=1))

    def __andPairwise(self, m1, m2):
        '''calculate AND by calling distance function for each pair of topics'''
        topics1, topics2 = m1.topicIds(), m2.topicIds()
        self.nearestTopic_, self.nearestDist_ = {}, {}
        avgmind = 0.0
        for i, t1 in enumerate(topics1):
            tvec1 = m1.topicVector(t1)
            mind, mint = None, None
            for t2 in topics2:
                d = self.dist(tvec1, m2.topicVector(t2))
                if mind is None or d < mind:
                    mind, mint = d, t2
            self.nearestDist_[t1] = mind
            self.nearestTopic_[t1] = mint
            avgmind += mind
        avgmind /= len(topics1)
        self.and_ = avgmind
        return avgmind

class TopicCoverDist(IdComposer):
    '''
    Measures number of topics in a TopicModel that are covered (have same topic) by another model.
    The criterion for sameness is based on a distance thresholding.
    '''

    def __init__(self, distance, threshold):
        '''
        :param distance: distance function on pairs of topic vectors
        '''
        self.distance = distance
        self.threshold = threshold
        if distance in supportMatrixCalc: self.pairwise = False
        else: self.pairwise = True
        IdComposer.__init__(self)
        self._and = AverageNearestDistance(distance, pairwise=self.pairwise)

    def __call__(self, target, model):
        self._and(target, model)
        if not self.pairwise:
            return float(sum(self._and.closestDist <= self.threshold)) / target.numTopics()
        else:
            nd = self._and.nearestDist_
            return float(sum(nd[t]<=self.threshold for t in nd))/target.numTopics()

def topicMatrix(m):
    tid = m.topicIds()
    W = len(m.topicVector(tid[0]))  # topic vector length
    mat = np.empty((len(tid), W))
    for i, ti in enumerate(tid):
        mat[i] = m.topicVector(ti)
    return mat

def printAndDetails(m1, m2, andist, printAlltopics=True):
    ''' Utility function that prints target topics and corresponding nearest topics. '''
    print 'model1: %d topics, %s' % (m1.numTopics(), m1.id)
    print 'model2: %d topics, %s' % (m2.numTopics(), m2.id)
    print 'AND: %.4f' % andist(m1, m2)
    if printAlltopics:
        topics = [ti for ti in m1.topicIds()]
        topics.sort(key=lambda ti: andist.nearestDist_[ti], reverse=True)
        for i, t1 in enumerate(topics):
            print '  ndist  [%2d]  %.4f' % (i, andist.nearestDist_[t1])
            print '  topic  [%2s]: %s' % (str(t1), m1.topic2string(t1, 20))
            nt = andist.nearestTopic_[t1]
            print '  ntopic [%2s]: %s' % (str(nt), m2.topic2string(nt, 20))

def testCoverDist():
    from pytopia.testing import setup
    from pytopia.measure.topic_distance import cosine as cosineDist
    from pytopia.context.ContextResolver import resolve
    models = resolve(*['model1', 'nmf_model1'])
    d = TopicCoverDist(cosineDist, 0.01)
    # when a model covers itself, all topics must be covered
    for m in models:
        assert d(m, m) == m.numTopics()

if __name__ == '__main__':
    testCoverDist()