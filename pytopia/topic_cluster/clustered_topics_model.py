from pytopia.topic_model.TopicModel import TopicModel, Topic
from pytopia.tools.IdComposer import IdComposer
from pytopia.resource.loadSave import pickleObject
from pytopia.context.ContextResolver import resolve, resolveId

import os, cPickle, numpy as np
from os import path

class ClusteredTopicsModel(IdComposer, TopicModel):
    '''
    Pytopia topic model created by clustering topics of other models.
    Each topic will correspond to a cluster of topics.
    '''

    def __init__(self, topics, topicsId, clusterer, aggregate='average'):
        '''
        :param topics: Topics to be clustered, a list of items each being either
                a Topic or a TopicModel or a folder (string path) to a saved topic model
        :param clusterer: TopicClusterer object encapsulating a clustering algorithm
        :param topicsId: serves for topic set identification, since currently id composition
                works by aggregating string ids of sub-objects and ids are used for saving
                objects to folders, it is infeasible to aggregate ids of many topics.
                This is a workaround until the method for creating ids is upgraded.
        :param aggregate: Method 'center' to use cluster centers the clusterer provides,
                or 'average' to average topics in a cluster
        '''
        self._topicSource = topics
        self.topicsId = topicsId
        self.clusterer = clusterer
        self.aggregate = aggregate
        IdComposer.__init__(self, ['topicsId', 'clusterer', 'aggregate'])

    def build(self):
        self.__createTopicList()
        self.__getDictionary()
        self._clusters = self.clusterer(self._topics)
        self.__createTopicData()

    def __createTopicList(self):
        ''' Create a list of all topics, loading models from folders in necessary. '''
        from pytopia.topic_cluster.utils import loadTopics
        #todo more validation of topics and models (vector length, dictionaries)
        self._topics = loadTopics(self._topicSource, checkDuplicates=True)

    def __getDictionary(self):
        '''
        Validate that all models of the topics being clustered have the same dictionary.
        Create self.dictionary property.
        '''
        modelIds = list(set(t.model for t in self._topics))
        dicts = set()
        for mid in modelIds:
            m = resolve(mid)
            dict = resolveId(m.dictionary)
            dicts.add(dict)
        if len(dicts) > 1:
            raise Exception('All the models must share dictionary. Models:\n%s'%
                            '\n'.join(resolve(mid) for mid in modelIds))
        else:
            self.dictionary = dicts.pop()

    def __createTopicData(self):
        ''' Construct topic matrix and other model-related data from clusters. '''
        # remove empty clusters
        self._clusters = [ cl for cl in self._clusters if len(cl) > 0 ]
        self.numClusters = len(self._clusters)
        # length of topic vector, take first topic, all topic vectors should be of same size
        M = len(self._clusters[0][0].vector)
        self._topicMatrix = np.ndarray(shape=(self.numClusters, M), dtype=np.float32)
        if self.aggregate == 'average':
            for i, cl in enumerate(self._clusters):
                vec = None
                for t in cl:
                    if vec is None: vec = t.vector.copy()
                    else: vec += t.vector
                vec /= len(cl)
                self._topicMatrix[i] = vec
        elif self.aggregate == 'center':
            for i, cl in enumerate(self._clusters):
                self._topicMatrix[i] = cl.center.vector
        else: raise Exception('unsupported topic aggregation method: %s' % self.aggregate)

    def topicIds(self): return range(self.numClusters)

    def numTopics(self): return self.numClusters

    def topicVector(self, topicId):
        ''':param topicId: integer index of a topic'''
        return self._topicMatrix[topicId]

    def __getstate__(self):
        return IdComposer.__getstate__(self), \
               self.dictionary, self.numClusters, self._topics, self._clusters

    topicMatrixFile = 'topicMatrix.npy'
    def save(self, folder):
        if not path.exists(folder): os.makedirs(folder)
        np.save(path.join(folder, self.topicMatrixFile), self._topicMatrix)
        pickleObject(self, folder)

    def __setstate__(self, state):
        idcs, self.dictionary, self.numClusters, self._topics, self._clusters = state
        IdComposer.__setstate__(self, idcs)

    def load(self, folder):
        self._topicMatrix = np.load(path.join(folder, self.topicMatrixFile))
        TopicModel.load(self, folder)

from pytopia.resource.ResourceBuilder import SelfbuildResourceBuilder
ClusteredTopicsModelBuilder = SelfbuildResourceBuilder(ClusteredTopicsModel)

