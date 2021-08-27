from pytopia.topic_model.TopicModel import TopicModel, Topic
from pytopia.tools.IdComposer import IdComposer, autoId
from pytopia.context.ContextResolver import resolveIds
from pytopia.topic_model.ArtifTopicModel import ArtifTopicModel

import numpy as np

class StackedTopicModel(TopicModel, IdComposer):

    def __init__(self, models, id=None, text2tokens=None):
        self.id = id if id is not None else autoId(models)
        self.text2tokens = text2tokens
        self._models = models
        self._checkModels()

    def _checkModels(self):
        '''
        Check that all models share dictionary and corpus.
        Add tihs dict and corpus as self properties.
        '''
        c, d = None, None
        for i, m in enumerate(self._models):
            mc, md = resolveIds(m.corpus, m.dictionary)
            if i == 0: c, d = mc, md
            else:
                err = None
                if c != mc: err = 'corpus mismatch: %s vs %s' % (c, mc)
                if d != md: err = 'dict mismatch: %s vs %s' % (d, md)
                if err: raise Exception(err)
        self.corpus, self.dictionary = c, d

    def build(self):
        # todo topic deduplication?
        topicMatrix = np.concatenate([m.topicMatrix() for m in self._models], axis=0)
        makeDocTopics = sum(m.corpusTopicVectors() is not None for m in self._models)
        if makeDocTopics:
            docTopicMatrix = np.concatenate([m.corpusTopicVectors() for m in self._models], axis=1)
        else: docTopicMatrix = None
        self._atm = ArtifTopicModel(topicMatrix, docTopicMatrix=docTopicMatrix)
        self.T = topicMatrix.shape[0]

    def topicIds(self): return range(self.T)

    def numTopics(self): return self.T

    def topic(self, topicId):
        return Topic(self, topicId, self.topicVector(topicId))

    def topicVector(self, topicId):
        if not hasattr(self, '_topicMatrix'): self._topicMatrix = self._atm.topicMatrix()
        return self._topicMatrix[topicId]

    def topicMatrix(self, dtype=None):
        if not hasattr(self, '_topicMatrix'): self._topicMatrix = self._atm.topicMatrix()
        return self._topicMatrix

    def corpusTopicVectors(self): return self._atm.corpusTopicVectors()
