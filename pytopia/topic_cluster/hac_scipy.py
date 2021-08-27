from pytopia.tools.IdComposer import IdComposer
from pytopia.topic_cluster.generic_functionality import TopicClusteringHelper

from scipy.cluster.hierarchy import average, fcluster, complete
from scipy.spatial.distance import squareform
import numpy as np

class HacScipyTopicCluster(IdComposer):
    '''
    Hierarchical agglomerative clustering of topic model topics using
    clustering functionality from scipy.
    '''

    def __init__(self, distance, linkage, numClusters):
        '''
        :param distance: distance function on topic vectors
        :param linkage: 'average' or 'complete'
        '''
        self.distance = distance
        self.linkage = linkage
        self.numClusters = numClusters
        IdComposer.__init__(self)

    def __call__(self, topics):
        h = TopicClusteringHelper(topics)
        h.createTopicMatrix()
        dist = self.distance(h.topicMatrix, h.topicMatrix)
        np.fill_diagonal(dist, 0.0)
        cdist = squareform(dist, checks=False)
        if self.linkage == 'average': clhier = average(cdist)
        elif self.linkage == 'complete': clhier = complete(cdist)
        else: raise Exception('unsupported linkage criterion: %s' % self.linkage)
        cl = fcluster(clhier, t=self.numClusters, criterion='maxclust')
        cl = cl - 1
        return h.clusteringFromLabels(cl)

from pytopia.measure.topic_distance import cosine as cosineDist
from pytopia.topic_cluster.testing import runTopicModelsTest

def basicTest():
    hac = HacScipyTopicCluster(cosineDist, 'average', 100)
    print hac.id

def test():
    basicTest()
    clusters1 = runTopicModelsTest(HacScipyTopicCluster, distance=cosineDist,
                                   linkage='average', numClusters=5)
    assert len(clusters1) == 5
    clusters2 = runTopicModelsTest(HacScipyTopicCluster, distance=cosineDist,
                                   linkage='average', numClusters=10)
    assert len(clusters2) == 10
    clusters3 = runTopicModelsTest(HacScipyTopicCluster, distance=cosineDist,
                                   linkage='complete', numClusters=7)
    assert len(clusters3) == 7

if __name__ == '__main__':
    test()

