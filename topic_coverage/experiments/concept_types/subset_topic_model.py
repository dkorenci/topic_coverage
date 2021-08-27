from pytopia.topic_model.TopicModel import TopicModel, Topic
from pytopia.tools.IdComposer import IdComposer
from pytopia.context.ContextResolver import resolve

class SubsetTopicModel(TopicModel, IdComposer):
    '''
    TopicModel that is a subset of another, ie its topics
    are a subset of another model's topics.
    '''

    def __init__(self, label, model, topics, origTopicIds=False, **params):
        '''
        :param model: original model this model will be a subset of
        :param topics: list of topicIds representing this model's topics
        '''
        self.supermodel = resolve(model)
        self.label = label
        self.corpus, self.dictionary, self.text2tokens = \
            model.corpus, model.dictionary, model.text2tokens
        self._origIds = origTopicIds
        # todo this can reorder given topics,
        # leave the ordering unchanged (and shift the responsibility for uniqueness to client code)?
        self._topics = list(set(topics))
        self._createTopicIndex()
        for k, v in params.iteritems(): self.__setattr__(k, v)
        IdComposer.__init__(self)

    def topicIds(self): return self._topics

    def numTopics(self): return len(self._topics)

    def topic(self, topicId):
        if self._origIds: return self.supermodel.topic(topicId)
        else: return TopicModel.topic(self, topicId)

    def topicVector(self, topicId): return self.supermodel.topicVector(topicId)

    def topicMatrix(self, dtype=None):
        if not hasattr(self, '_tmatrix'):
            mtx = self.supermodel.topicMatrix()
            topicInds = [self._tind[tid] for tid in self._topics]
            self._tmatrix = mtx[topicInds]
        return self._tmatrix

    def _createTopicIndex(self):
        ''' Create map topicId -> ordinal of the topic in the supermodel '''
        supind = { tid : ti for ti, tid in enumerate(self.supermodel.topicIds()) }
        self._tind = { tid : supind[tid] for tid in self._topics }

    def corpusTopicVectors(self, txtId=None):
        #todo optimize by storing corpus-topic matrix locally
        topicInds = [self._tind[tid] for tid in self._topics]
        if txtId:
            ctv = self.supermodel.corpusTopicVectors(txtId)
            return ctv[topicInds]
        else:
            ctm = self.supermodel.corpusTopicVectors()
            return ctm[:, topicInds]

