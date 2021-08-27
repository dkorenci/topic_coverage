from pytopia.context.ContextResolver import resolve
from pytopia.tools.IdComposer import IdComposer
from pytopia.utils.print_ import topTopicWords

from math import log

class TfidfCoherence(IdComposer):
    '''
    Calculates topic coherence after the method in the article:
     Topic modelling for qualitative studies [1]
     Coherence is average pairwise word similarity
     where similarity is based on word-doc tfidf scores.
    '''

    def __init__(self, topWords, epsilon=1e-7, cached=True):
        '''
        :param topWords: number of top topic words to consider
        :param epsilon: smoothing constant to add to calculated word pair similarity
        '''
        self.topWords = topWords
        self.epsilon = epsilon
        IdComposer.__init__(self)
        self.cached = cached
        if cached: self.__cache = {}

    # requires: worddoc_index_builder
    def __call__(self, topic):
        '''
        :param topic: (modelId, topicId)
        '''
        mid, tid = topic
        wdib = resolve('worddoc_index_builder')
        model = resolve(mid)
        wdi = wdib(corpus=model.corpus, text2tokens=model.text2tokens,
                   dictionary=model.dictionary)
        words = model.topTopicWords(tid, self.topWords)
        # sum coherences of word pairs
        W = len(words); numPairs = 0
        score = 0.0
        for i, wi in enumerate(words):
            for j in xrange(i+1, len(words)):
                numPairs += 1; wj = words[j]
                score += self.__tfidfCoherence(wi, wj, wdi)
        return score/numPairs

    def __tfidfCoherence(self, w1, w2, wdi):
        '''
        Calculate tf-idf based word similarity, after equation (5) in [1].
        :param w1, w2: string words
        :param wdi: worddoc_index
        '''
        if self.cached:
            key = 'wordsim_%s_%s_%s' % (w1, w2, wdi.id)
            if key in self.__cache: return self.__cache[key]
        # for words, get textId:wordCnt maps
        docs1, docs2 = dict(wdi.wordDocs(w1)), dict(wdi.wordDocs(w2))
        # calculate denominator
        dn = 0.0
        for txtId, wordCnt in docs1.iteritems():
            dn += self.tfidf(w1, txtId, wdi,
                             wordDocs=len(docs1), wordDocFreq=wordCnt)
        # calculate numerator
        if len(docs1) > len(docs2):
            docs1, docs2 = docs2, docs1
            w1, w2 = w2, w1
        nm = self.epsilon
        for txtId, wordCnt in docs1.iteritems():
            if txtId in docs2:
                nm += self.tfidf(w1, txtId, wdi, wordDocs=len(docs1), wordDocFreq=wordCnt) * \
                      self.tfidf(w2, txtId, wdi, wordDocs=len(docs2), wordDocFreq=docs2[txtId])
        sim = log((nm+self.epsilon)/dn)
        if self.cached: self.__cache[key] = sim
        return sim

    def tfidf(self, w, txtId, wdi, wordDocs=None, wordDocFreq=None):
        '''
        Calculate tfidf core for a word and a text, after equation (6) in [1].
        :param wdi: worddoc_index
        :param wordDocs: number of documents containing word w, if None it is calculated
        :param wordDocFreq: number of times w occurs in txtId, if None it is calculated
        '''
        if self.cached:
            key = 'tfidf_%s_%s_%s' % (w, txtId, wdi.id)
            if key in self.__cache: return self.__cache[key]
        if wordDocs is None or wordDocFreq is None:
            wdocs = wdi.wordDocs(w)
            if wordDocs is None: wordDocs = len(wdocs)
            if wordDocFreq is None: wordDocFreq = dict(wdocs)[txtId]
        idf = log(float(wdi.numDocs())/wordDocs)
        maxFreq = max(cnt for word, cnt in  wdi.docWords(txtId))
        tf = (0.5 + wordDocFreq/float(maxFreq))
        res = tf*idf
        if self.cached: self.__cache[key] = res
        return res
