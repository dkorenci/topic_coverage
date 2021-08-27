from pytopia.context.ContextResolver import resolve
from pytopia.tools.IdComposer import IdComposer

from pytopia.evaluation.stability.hungarian import Hungarian
from pytopia.topic_model.utils import corpusTopicWeights
from scipy.spatial.distance import cdist

import numpy as np

class ModelmatchBipartite(IdComposer):
    '''
    Match two topic models using hungarian algorithm to find max. bipartite matching of topics.
    Matching is based on a function calculating matching score between two topics.
    '''

    def __init__(self, topicMatch):
        self.topicMatch = topicMatch
        IdComposer.__init__(self)

    def __call__(self, m1, m2):
        m1, m2 = resolve(m1, m2)
        # build score matrix
        T1, T2 = m1.numTopics(), m2.numTopics()
        if self.topicMatch == 'word-cosine':
            mx1, mx2 = m1.topicMatrix(), m2.topicMatrix()
            matchMatrix = 1.0 - cdist(mx1, mx2, 'cosine')
            assert matchMatrix.shape == (T1, T2)
        else:
            matchMatrix = np.zeros((T1, T2), np.float32)
            for i, t1 in enumerate(m1):
                for j, t2 in enumerate(m2):
                    matchMatrix[i, j] = self.topicMatch(t1, t2)
        # apply hungarian matching
        h = Hungarian()
        costMatrix = h.make_cost_matrix(matchMatrix)
        h.calculate(costMatrix)
        results = h.get_results()
        # compute score based on similarities
        score = 0.0
        for (row, col) in results:
            score += matchMatrix[row, col]
        score /= len(results)
        return score

class TopicmatchVectorsim(IdComposer):
    ''' Matching score between Topics using a similarity measure of
    either topic-word or topic-document vectors. '''

    def __init__(self, vecSim, vectors='word'):
        '''
        :param vecSim: similarity score for 2 numpy vectors
        :param vectors: 'word' or 'doc'
        '''
        self.vecSim = vecSim
        self.vectors = vectors
        IdComposer.__init__(self)

    def __call__(self, t1, t2):
        if self.vectors == 'word': return self.vecSim(t1.vector, t2.vector)
        elif self.vectors == 'doc':
            return self.vecSim(corpusTopicWeights(t1), corpusTopicWeights(t2))
        else: raise Exception('vectors must be either "word" or "doc"')

