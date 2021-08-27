import numpy as np

from pytopia.resource.tools import tfIdfMatrix
from pytopia.tools.IdComposer import IdComposer
from pytopia.topic_functions.tools import cached_function


@cached_function
class TfidfVarianceCoherence(IdComposer):
    '''
    Calculates coherence as variance of tfidf vectors of top topic-related texts.
    '''

    def __init__(self, threshold=100):
        self.threshold = threshold
        IdComposer.__init__(self)

    # requires corpus_topic_index_builder
    # requires corpus_tfidf_builder
    def __call__(self, topic):
        '''
        :param topic: (modelId, topicId)
        :return:
        '''
        m = tfIdfMatrix(topic, self.threshold)
        N = m.shape[0] # num. rows
        a = np.average(m, axis=0)
        m -= a; m *= m
        v = (m.sum(axis=1).sum()) / N
        return -v
