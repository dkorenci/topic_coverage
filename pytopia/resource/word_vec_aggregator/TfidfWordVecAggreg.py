from pytopia.context.ContextResolver import resolve, resolveIds
from pytopia.tools.IdComposer import IdComposer

import gc

# requires corpus_tfidf_builder
class TfidfWordVecAggreg(IdComposer):
    '''
    Creates vectors from texts by aggregating word vectors weighted with tfidf scores.
    '''

    # todo implement automatic dictionary creation, via builder
    def __init__(self, text2tokens, word2vector, corpus, dictionary):
        self.text2tokens, self.word2vector = text2tokens, word2vector
        self.corpus, self.dictionary = corpus, dictionary
        IdComposer.__init__(self)
        self.__tfidfBuilt = False
        self.__word2vec = resolve(word2vector)

    def __call__(self, txto):
        self.__buildCorpusTfidf()
        from scipy.sparse import dok_matrix
        textId = resolveIds(txto)
        vec = dok_matrix(self.__ctf.tfidf(textId, format='sparse'))
        d = resolve(self.dictionary)
        res = None
        for loc, val in vec.items():
            word = d.index2token(loc[1])
            vector = self.__word2vec(word)
            if vector is not None:
                if res is None: res = val*(vector.copy())
                else: res += val*vector
            else: pass
        return res

    # requires corpus_tfidf_builder
    def __buildCorpusTfidf(self):
        if self.__tfidfBuilt: return
        builder = resolve('corpus_tfidf_builder')
        self.__ctf = builder(corpus=self.corpus, text2tokens=self.text2tokens,
                             dictionary=self.dictionary)
        self.__tfidfBuilt = True

