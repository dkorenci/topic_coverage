from pytopia.context.ContextResolver import resolve
from pytopia.resource.tools import tfIdfMatrix
from pytopia.measure.topic_distance import l2, cosine

import numpy as np

class MatrixNormCoherence():
    '''
    Calculates coherence as norm of matrix of distances of tfidf vectors of top texts.
    '''

    @property
    def id(self): return 'matrix_norm_coherence'

    def __init__(self, dist):
        '''
        :param dist: function for calculating distance between two vectors/matrices
        '''
        self.__dist = dist

    # requires corpus_topic_index_builder
    # requires corpus_tfidf_builder
    def __call__(self, topic):
        '''
        :param topic: (modelId, topicId)
        :return:
        '''
        m = tfIdfMatrix(topic, 100)
        N = m.shape[0]  # num. rows
        dm = self.__dist(m, m)
        d = np.average(dm)
        return -d
