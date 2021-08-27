from pytopia.tools.IdComposer import IdComposer
from pytopia.resource.tools import tfIdfMatrix
from pytopia.measure.topic_distance import cosine

import networkx as nx

class GraphCCCoherence(IdComposer):
    '''
    Calculates coherence clustering coefficient of top topic
    documents, where connections are based on distances of tf-idf vectors.
    '''

    def __init__(self, threshold):
        self.threshold = threshold
        IdComposer.__init__(self)

    # requires corpus_topic_index_builder
    # requires corpus_tfidf_builder
    def __call__(self, topic):
        '''
        :param topic: (modelId, topicId)
        :return:
        '''
        m = tfIdfMatrix(topic, 100)
        N = m.shape[0] # num docs
        distM = cosine(m, m)
        g = nx.Graph()
        for i in xrange(N):
            for j in xrange(i+1, N):
                if distM[i, j] <= self.threshold:
                    g.add_edge(i, j)
        return nx.transitivity(g)
