'''
Generic functionality for all the topic clusterers.
'''

from pytopia.context.ContextResolver import resolve, resolveId

import numpy as np

class TopicClusterer():
    '''
    Class containing interface methods for topic clusterers,
     to be used as a reference
    Each topic if represented as (modelId, topicId) pair.
    '''

    def __call__(self, topics):
        '''
        :param topics: Topics to be clustered, list like of
         either Topic and TopicModel objects, possibly mixed.
         For a model, all the model's topic will be clustered.
        :return: list of Clusters
        '''
        raise NotImplementedError

    def add(self, topics):
        '''
        Optional, for clusterers supporting cumulative clustering.
        :param topics: list of Topic and TopicModel objects
        :return: list of Clusters
        '''
        raise NotImplementedError

    def clustering(self):
        '''
        Return current clustering of objects.
        :return: list of Clusters
        '''


class Cluster(list):
    '''
    List-like of items with additional properties
    '''

    def __init__(self, elements=None, center=None, **properties):
        '''
        :param properties: additional cluster properties
        '''
        list.__init__(self)
        if elements: self.extend(elements)
        self.center = center
        for name, value in properties.iteritems():
            self.__dict__[name] = value

class TopicClusteringHelper():
    '''
     Generic data structures and operations for topic clustering for topic clusterers.
     Initialized with a list of topics and/or models, it creates a duplicate-free
     list of all topics and validates all the topic vectors are of same length.
     Creates topic-related matrices and transforms clustering results from
        sklearn clustering output to list of Cluster objects.
    '''

    def __init__(self, topics):
        self._topics = []
        self.__addTopics(topics)

    @property
    def topics(self): return list(self._topics)

    @property
    def numTopics(self): return self._numTopics

    @property
    def topicMatrix(self): return self._topicMatrix

    @property
    def vectorLength(self):
        '''

        :return:
        '''
        return self._vectorLength

    def __addTopics(self, topics):
        ''' Add standalone Topics and Topics from models to data structs. '''
        from pytopia.topic_model.TopicModel import isTopicModel
        for t in topics:
            if isTopicModel(t): self.__addModel(t)
            else: self.__addTopic(t)
        self._numTopics = len(self._topics)
        vecLens = set(len(t.vector) for t in self._topics)
        if len(vecLens) > 1:
            raise Exception('All topic must have vectors of the same length')
        self._vectorLength = vecLens.pop()

    def __addModel(self, m):
        for tid in m.topicIds(): self.__addTopic(m.topic(tid))

    def __addTopic(self, t):
        #todo check for duplicates
        self._topics.append(t)

    def createTopicMatrix(self):
        '''
        Create ndarray containing vectors of all topics, with shape (numTopics, topicSize)
        '''
        self._topicMatrix = np.ndarray((self._numTopics, self._vectorLength))
        for ti, top in enumerate(self._topics):
            self._topicMatrix[ti] = top.vector
        return self._topicMatrix

    def topicXtopic(self, f):
        '''
        Create numTopics x numTopics distance matrix
        :param f: function on pairs topic vectors
        :return:
        '''
        # todo run f on matrices, if supported
        m = np.empty((self._numTopics, self._numTopics))
        for i, ti in enumerate(self._topics):
            m[i,i] = 0.0
            for j, tj in enumerate(self._topics):
                if i != j: m[i, j] = f(ti.vector, tj.vector)
        return m

    # todo pull this out as stand-alone generic method
    def clusteringFromScikit(self, skc, noClustLabel=-1, centerIndices='cluster_centers_indices_'):
        '''
        Create TopicClustering object from fitted sciki-learn clusterer (skc).
        skc must have attribute labels_, array of cluster indexes for the data (topics),
         this array can contain noClustLabel value that indicates no cluster assignment.
        :param skc: fitted scikit-learn clusterer
        :param centerIndices: name of the attribute containing indices of cluster centers,
            or None if there is no such attribute.
        '''
        # print skc
        # print skc.labels_
        # print skc.cluster_centers_indices_
        # todo put all data validation in one place
        # array containing at index i, cluster label (in [0..numClusters-1]) for topic i
        clusterLabels = skc.labels_
        # set of all cluster labels indicating a valid cluster
        if hasattr(skc, 'cluster_centers_'):
            # todo use cluster_centers_ if cluster_centers_indices_ are not defined ?
            numClust = len(skc.cluster_centers_)
        elif centerIndices and hasattr(skc, centerIndices):
            numClust = len(getattr(skc, centerIndices))
        else:
            numClust = len(set(l for l in clusterLabels if l != noClustLabel))
        distinctLabels = set(l for l in clusterLabels if l != noClustLabel)
        assert numClust == len(distinctLabels)
        if centerIndices and hasattr(skc, centerIndices): cind = getattr(skc, centerIndices)
        else: cind = None
        return self.clusteringFromLabels(clusterLabels, noClustLabel, cind)

    def clusteringFromLabels(self, labels, noClustLabel=-1, centerIndices=None):
        '''
        Form topic Clusters from a list of topics' cluster labels and
            indices of center topics
        :param labels: array with self.numTopics length, with cluster labels for each
                topic, labels must be all numbers in range [0 .. numClust-1]
        :param centerIndices: list of length numClust, containing indices of
                topics that are cluster centers, each in range [0 .. self.numTopics]
        :param noClustLabel: index indicating that a topic is in no cluster
        :return: list of Cluster objects containing topics
        '''
        assert len(labels) == self._numTopics
        distinctLabels = set(l for l in labels if l != noClustLabel)
        numClust = len(set(l for l in labels if l != noClustLabel))
        # check that cluster labels are equal to [0,1, ... , numClust-1]
        assert set(range(numClust)) == distinctLabels
        clusters = [ Cluster() for _ in range(numClust) ]
        for i, l in enumerate(labels):
            if l == noClustLabel: continue
            clusters[l].append(self._topics[i])
        if centerIndices is not None:
            for i, ci in enumerate(centerIndices):
                clusters[i].center = self._topics[ci]
        return clusters
