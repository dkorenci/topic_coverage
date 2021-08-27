from pytopia.topic_model.TopicModel import Topic, TopicModel
from pytopia.tools.IdComposer import IdComposer
from pytopia.context.ContextResolver import resolveIds
from pytopia.resource.loadSave import pickleObject

import numpy as np, os, pickle
from os import path

class ArtifTopicModel(TopicModel, IdComposer):
    '''
    Artificial topic model defined by a matrix of _topics.
    '''

    def __init__(self, topicMatrix, corpus=None, dictionary=None, text2tokens=None,
                       docTopicMatrix=None):
        '''
        :param topicMatrix: a 2d ndarray
        :param dictionary: mapping of words (string tokens) to integer indices
        '''
        self._topics = topicMatrix
        self._numTopics = self._topics.shape[0]
        self._docTopics = docTopicMatrix
        self.corpus, self.dictionary, self.text2tokens = \
            resolveIds(corpus, dictionary, text2tokens)
        IdComposer.__init__(self)

    def numTopics(self): return self._numTopics

    def topicVector(self, tid): return self._topics[tid]

    def topicIds(self): return range(self._numTopics)

    def corpusTopicVectors(self):
        return self._docTopics

    def build(self): pass # just to comply to buildable resource interface

    def __matrixFile(self, folder): return path.join(folder, 'topicMatrix.npy')

    def __docTopicsFile(self, folder): return path.join(folder, 'docTopicMatrix.npy')

    def save(self, folder):
        '''np.save topic and doc-topic matrix and than pickle self'''
        pickleObject(self, folder)
        np.save(self.__matrixFile(folder), self._topics)
        if self._docTopics is not None:
            np.save(self.__docTopicsFile(folder), self._docTopics)

    def load(self, folder):
        '''load (non-pickled) topic matrix'''
        self._topics = np.load(self.__matrixFile(folder))
        self._numTopics = self._topics.shape[0]
        if path.exists(self.__docTopicsFile(folder)):
            self._docTopics = np.load(self.__docTopicsFile(folder))
        else: self._docTopics = None

from pytopia.resource.ResourceBuilder import SelfbuildResourceBuilder
ArtifTopicModelBuilder = SelfbuildResourceBuilder(ArtifTopicModel)