from pytopia.context.ContextResolver import resolve, resolveIds
from pytopia.tools.IdComposer import IdComposer, deduceId

#TODO tests
class WordvecDistCoherence(IdComposer):
    '''
    Given word vectorizer, calculates topic coherence as average
    pairwise vector similarity across all (top) word pairs.
    '''

    def __init__(self, word2vector, distance, inverseToken = None, topWords=10):
        '''
        :param word2vector: callable mapping word 2 vector or None it there's no mapping
        :param distance: distance function on vector
        :param inverseToken: if not None, mapping token->word that will be applied
                to top topic words before mapping to vectors
        :param topWords: number of top words per topic to use
        '''
        self.word2vector, self.inverseToken = resolveIds(word2vector, inverseToken)
        self.__w2v, self.__invTok= word2vector, inverseToken
        self.distance = deduceId(distance); self.__dist = distance
        self.topWords = topWords
        IdComposer.__init__(self)

    def __call__(self, topic):
        '''
        :param topic: (modelId, topicId)
        :return:
        '''
        mid, tid = topic
        model = resolve(mid)
        words = model.topTopicWords(tid, topw=self.topWords)
        if self.inverseToken is not None:
            invToks = []
            for w in words:
                inv = self.__invTok(w)
                #print w, inv
                if inv is not None: invToks.append(inv)
                else: invToks.append(w)
            words = invToks
        vecs = []
        for w in words:
            v = self.__w2v(w)
            if v is not None:
                #print w, ','.join('%.4f' % e for e in v)
                vecs.append(v)
        numPairs = 0
        avg = 0.0
        for i, v1 in enumerate(vecs):
            for j in range(i+1, len(vecs)):
                v2 = vecs[j]
                numPairs += 1
                avg += self.__dist(v1, v2)
        return - avg / numPairs


