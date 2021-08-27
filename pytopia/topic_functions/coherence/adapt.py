from pytopia.context.ContextResolver import resolve

class WordsStringCohAdapter():
    '''
    Adapts coherence measure that accepts a string of
    whitespace separated words to accept pytopia model topics.
    '''

    def __init__(self, coh, topW, id=None):
        '''
        :param coh: adapted coherence measure
        :param topW: number of top topic words to use for coherence
        '''
        self.__coh = coh
        self.__topw = topW
        self.__id = id

    @property
    def id(self): return self.__id

    def __call__(self, topic):
        '''
        :param topic: (modelId, topicId)
        :return:
        '''
        mid, tid = topic; model = resolve(mid)
        return self.__coh(model.topic2string(tid, topw=self.__topw))
