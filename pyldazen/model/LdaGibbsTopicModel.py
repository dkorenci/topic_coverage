from pytopia.topic_model.TopicModel import TopicModel, Topic
from pytopia.tools.IdComposer import IdComposer
from pytopia.context.ContextResolver import resolveIds, resolve
from pytopia.resource.loadSave import pickleObject
from pytopia.resource.ResourceBuilder import SelfbuildResourceBuilder

import os, numpy as np
from os import path

from pyldazen.build.LdaGibbsInferer import LdaGibbsInferer

class LdaGibbsTopicModel(TopicModel, IdComposer):

    def __init__(self, corpus, dictionary, text2tokens,
                        numTopics, alpha=None, beta=0.01, gibbsIter=1000,
                        fixedTopics=None, idLabel=None, rndSeed=88911):
        '''
        LDA Topic Model with gibbs sampling inference, supports fixed topics.
        Newly inferred topic are indexed  with 0..numTopics-1,
        and fixed topics numTopics..numTopics+numFixed-1.
        :param alpha: float, prior for symmetric dirichlet document-topic prior
        :param beta: float or matrix
        :param gibbsIter: number of gibbs sampling iterations to run when building model
        :param fixedTopics: ndarray of topics with shape (numFixed, dictionary.maxIndex+1)
        :param idLabel: additional id-info describing beta (if matrix) and fixedTopics
        '''
        #todo is dictionary necessary? default it to None?
        self.corpus, self.dictionary, self.text2tokens = \
            resolveIds(corpus, dictionary, text2tokens)
        self.T, self.gibbsIter, self.rseed = numTopics, gibbsIter, rndSeed
        self._fixedTopics = fixedTopics
        self._numFixed = fixedTopics.shape[0] if fixedTopics is not None else 0
        self.idLabel = idLabel
        self.alpha = 50.0 / self.numTopics() if alpha is None else alpha
        atts = ['corpus', 'dictionary', 'text2tokens',
                'T', 'alpha', 'gibbsIter', 'rseed', 'idLabel']
        if isinstance(beta, (float, int)): # if beta is not a matrix, include in attributes
            self.beta = float(beta)
            atts.append('beta')
        IdComposer.__init__(self, atts)
        if not isinstance(beta, (float, int)): self.beta = beta

    def numTopics(self): return self.T + self._numFixed

    def topicIds(self): return range(self.numTopics())

    def topicVector(self, topicId):
        if self._numFixed > 0:
            if topicId < self.T: return self._topics[topicId]
            else: return self._fixedTopics[topicId-self.T]
        else: return self._topics[topicId]

    def fixedTopics(self):
        if self._fixedTopics is None: return None
        return [ self.topic(tid) for tid in range(self.T, self.numTopics()) ]

    def corpusTopicVectors(self):
        return self._doctopics

    #requires bow_corpus_builder
    def build(self):
        print 'Building', self.id
        dictionary = resolve(self.dictionary)
        bowCorpus = resolve('bow_corpus_builder')(self.corpus, self.text2tokens, self.dictionary)
        inferer = LdaGibbsInferer(self.T, self.alpha, self.beta, bowCorpus,
                                  self._numFixed, self._fixedTopics,
                                  maxWordIndex=dictionary.maxIndex())
        inferer.startInference()
        print 'running inference'
        inferer.runInference(self.gibbsIter)
        self._topics = inferer.calcTopicMatrix()
        if self._numFixed > 0: # check that topics are of same length
            assert self._topics.shape[1] == self._fixedTopics.shape[1]
        self._doctopics = inferer.calcDocTopicMatrix()
        inferer.finishInference()

    def __topicsFile(self, folder): return path.join(folder, 'topic.npy')
    def __fixedTopicsFile(self, folder): return path.join(folder, 'fixedTopics.npy')
    def __docTopicsFile(self, folder): return path.join(folder, 'docTopics.npy')

    def __getstate__(self):
        return IdComposer.__getstate__(self), self._numFixed

    def __setstate__(self, state):
        IdComposer.__setstate__(self, state[0])
        self._numFixed = state[1]

    def save(self, folder):
        '''np.save topic matrix and than pickle self'''
        if not path.exists(folder): os.makedirs(folder)
        np.save(self.__topicsFile(folder), self._topics)
        np.save(self.__docTopicsFile(folder), self._doctopics)
        if self._numFixed > 0: np.save(self.__fixedTopicsFile(folder), self._fixedTopics)
        pickleObject(self, folder)

    def load(self, folder):
        '''load (non-pickled) topic matrix'''
        self._topics = np.load(self.__topicsFile(folder))
        self._doctopics = np.load(self.__docTopicsFile(folder))
        if self._numFixed > 0:
            self._fixedTopics = np.load(self.__fixedTopicsFile(folder))

LdaGibbsTopicModelBuilder = SelfbuildResourceBuilder(LdaGibbsTopicModel)