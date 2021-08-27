from pytopia.context.ContextResolver import resolve
from pytopia.tools.IdComposer import IdComposer

class TopWordSelector(IdComposer):
    '''
    For a topic model topic, select top-K words
    or words with topic-weights above the threshold.
    '''

    def __init__(self, threshold):
        '''
        :param threshold: integer (for top K selction),
            a number between 0 and 1 (as topic-weight threshold)
        '''
        self.threshold = threshold2str(threshold)
        IdComposer.__init__(self)
        # todo check threshold value
        self.__threshold = threshold

    def __call__(self, topic):
        '''
        :param topic: (modelId, topicId)
        :return: iterable of top word indices
        '''
        from pytopia.utils.print_ import topVectorIndices
        import numpy as np
        mid, tid = topic
        model = resolve(mid)
        if 0.0 < self.__threshold < 1.0:
            vec = model.topicVector(tid)
            ind = np.nonzero(vec > self.__threshold)[0]
        elif isinstance(self.__threshold, (int, long)):
            ind = topVectorIndices(model.topicVector(tid), self.__threshold)
        d = resolve(model.dictionary)
        if d is None: return ind
        else: return [ d.index2token(i) for i in ind ]

def threshold2str(t):
    if isinstance(t, float): return '%.3f' % t
    else: return str(t)
