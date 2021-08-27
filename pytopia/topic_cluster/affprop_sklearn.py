from pytopia.tools.IdComposer import IdComposer
from pytopia.topic_cluster.generic_functionality import TopicClusteringHelper
from sklearn.cluster.affinity_propagation_ import AffinityPropagation

import numpy as np

#todo: unit tests
class AffpropSklearnTopicCluster(IdComposer):
    '''
    Cluster model topics using affinity propagation
    '''

    def __init__(self, similarity, preference=None):
        '''
        :param similarity: similarity function on pair of scipy vectors.
        :param preference: quantile of all the similarity values to be used as preference
                            hyperparameter, f.e. 0.5 means to take the median
        '''
        self.similarity = similarity
        self.preference = preference
        IdComposer.__init__(self)

    # def __getstate__(self): return self.similarity, self.preference
    # def __setstate__(self, state):
    #     self.__init__(state[0], state[1])

    def __call__(self, topics):
        self.helper = TopicClusteringHelper(topics)
        am, pref = self.__affinityAndPref()
        apc = AffinityPropagation(affinity='precomputed', preference=pref)
        apc.fit(am)
        return self.helper.clusteringFromScikit(apc)

    def __affinityAndPref(self):
        '''
        Create matrix of affinities based on similarity function.
         Create array of topic preferences based on self.preference.
         '''
        self.helper.createTopicMatrix()
        am = self.similarity(self.helper.topicMatrix, self.helper.topicMatrix)
        if self.preference is not None:
            n = am.shape[0]
            affs = np.array([am[i, j] for i in range(n) for j in range(n) if i != j])
            if self.preference < 1.0: # calc preference as percentile of affinities
                pref = np.percentile(affs, self.preference*100)
            else: # calc preference as self.preference * mean. affinity
                mean = np.mean(affs)
                if mean < 0: pref = mean/self.preference
                else: pref = mean*self.preference
            pref = np.repeat(pref, n)
        else: pref = None
        return am, pref

from pytopia.measure.topic_similarity import cosine as cosineSim

def basicTest():
    apc = AffpropSklearnTopicCluster(cosineSim)
    print apc.id

def models4testing():
    params = [
        {'similarity':cosineSim, 'preference':0.5},
        {'similarity':cosineSim, 'preference':0.8}
    ]
    return [AffpropSklearnTopicCluster(**par) for par in params]

def test():
    from pytopia.topic_cluster.testing import runTopicModelsTest
    basicTest()
    clusters1 = runTopicModelsTest(AffpropSklearnTopicCluster, similarity=cosineSim, preference=0.5)
    clusters2 = runTopicModelsTest(AffpropSklearnTopicCluster, similarity=cosineSim, preference=0.8)
    assert len(clusters2) > len(clusters1)

if __name__ == '__main__':
    test()
