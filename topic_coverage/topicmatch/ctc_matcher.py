from pytopia.tools.IdComposer import IdComposer
from pytopia.context.ContextResolver import resolve
from pytopia.measure.topic_distance import supportsBatch

import numpy as np

def modelDistMatrix(m1, m2, distance, batch=None):
    '''
    Return matrix of topic-topic distances for two TopicModels.
    :param m1, m2: TopicModels or ids
    :param m2:
    :param distance:
    :param batch:
    :return:
    '''
    m1, m2 = resolve(m1, m2)
    mx1, mx2 = m1.topicMatrix(), m2.topicMatrix()
    if batch is None: batch = supportsBatch(distance)
    if batch: return distance(mx1, mx2)
    else:
        mxd = np.empty((m1.numTopics(), m2.numTopics()), dtype=np.float32)
        for i in range(len(mx1)):
            for j in range(len(mx2)):
                mxd[i, j] = distance(mx1[i], mx2[j])
        return mxd

class CtcModelCoverage(IdComposer):
    '''
    Coverage-threshold curve matcher, matches two models by calculating distance-based
    coverages for various distance thresholds (two topics are same if distance is below threshold)
    and calculating as coverage measure the area under curve.
    '''

    def __init__(self, distance, min, max, intervals, batch=None):
        '''
        :param distance: distance function between topic vectors
        :param min: start of threshold intervals
        :param max: end of threshold intervals
        :param intervals: number of threshold intervals
        :param batch: it true, distance can compute in "batch" mode, on entire matrices
        '''
        self.distance = distance
        self.min, self.max, self.intervals, self.batch = min, max, intervals, batch
        IdComposer.__init__(self)

    def __call__(self, refmodel, model):
        # calc closest dist from model for each refmodel topic
        distMatrix = modelDistMatrix(refmodel, model, self.distance, self.batch)
        closestDist = np.min(distMatrix, axis=1)
        # calculate coverage ration for each threshold
        thresholds = np.linspace(self.min, self.max, self.intervals)
        cov = [None] * len(thresholds); refTopics = float(refmodel.numTopics())
        for i, th in enumerate(thresholds):
            covered = sum(closestDist <= th)
            cov[i] = covered/refTopics
        # calculate area under the curve, approx. as trapeze areas
        area = 0.0
        for i in range(1, len(thresholds)):
            area += (cov[i - 1] + cov[i]) * (thresholds[i] - thresholds[i - 1]) / 2.0
        return area

