from pytopia.resource.tools import tfIdfMatrix
from pytopia.tools.IdComposer import IdComposer

class ClusteringMixedness(IdComposer):
    '''
    Calculates mixedness by clustering points and calculating
    a specified metric of clustering quality.
    '''

    def __init__(self, clusterer, score):
        self.clusterer = clusterer
        self.score = score
        IdComposer.__init__(self)

    def __call__(self, topic):
        '''
        :param topic: (modelId, topicId)
        :return:
        '''
        m = tfIdfMatrix(topic, 100)
        self.clusterer.fit(m)
        return self.score(m, self.clusterer.labels_)
