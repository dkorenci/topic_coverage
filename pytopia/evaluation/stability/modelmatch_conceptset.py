from pytopia.context.ContextResolver import resolve, resolveId
from pytopia.tools.IdComposer import IdComposer

from pytopia.evaluation.stability.hungarian import Hungarian
from pytopia.topic_model.utils import corpusTopicWeights

import numpy as np

class ModelmatchRelConceptset(IdComposer):
    '''
    Match two topic models using a set of reference topics.
    Match is calculated as either relative or absolute number of matching ref. topics.
    '''

    def __init__(self, refmodel, topicMatch, relative=True):
        self.topicMatch = topicMatch
        self.refmodel = resolveId(refmodel)
        self.relative = relative
        IdComposer.__init__(self)

    def __call__(self, m1, m2):
        m1, m2, refmodel = resolve(m1, m2, self.refmodel)
        # build score matrix
        T1, T2 = m1.numTopics(), m2.numTopics()
        cset1, cset2 = coveredTopics(refmodel, m1, self.topicMatch), \
                       coveredTopics(refmodel, m2, self.topicMatch)
        numMatches = len(cset1.intersection(cset2))
        score = float(numMatches) / min(T1, T2)
        return score

def coveredTopics(refmodel, model, matcher):
    ''' Calculates and returns a set of ids of covered refmodel topics. '''
    cov = set()
    for rt in refmodel:
        covered = False
        for t in model:
            if matcher(rt, t):
                covered = True
                break
        if covered: cov.add(rt.topicId)
    return cov

