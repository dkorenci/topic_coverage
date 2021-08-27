from pytopia.context.ContextResolver import resolve, resolveIds
from pytopia.tools.IdComposer import IdComposer

from pytopia.resource.esa_vectorizer.EsaWordVectorizer import EsaWordVectorizer

# requires corpus_tfidf_builder
class EsaVectorizer(IdComposer):
    '''
    Creates vectors from texts by aggregating wikipedia concept vectors of words,
     using the method from the article:
     Computing Semantic Relatedness using Wikipedia-based Explicit Semantic Analysis
    '''

    def __init__(self, corpus, text2tokens, dictionary, javaVectors=False):
        self.corpus, self.text2tokens, self.dictionary = \
            resolveIds(corpus, text2tokens, dictionary)
        IdComposer.__init__(self)
        self.__word2vec = EsaWordVectorizer(javaVectors=javaVectors)
        self.__javaVectors = javaVectors
        self.__tfidfBuilt = False

    def __call__(self, txto):
        #print 'VECTORIZING TEXT'
        self.__buildCorpusTfidf()
        from scipy.sparse import dok_matrix
        textId = resolveIds(txto)
        vec = dok_matrix(self.__ctf.tfidf(textId, format='sparse'))
        d = resolve(self.dictionary)
        res = None
        for loc, val in vec.items():
            word = d.index2token(loc[1])
            #print word, val
            vector = self.__word2vec(word)
            if vector is not None:
                if res is None:
                    if self.__javaVectors:
                        res = vector.clone()
                        #print type(res), type(val)
                        res.multiply(float(val))
                    else: res = val*(vector.copy())
                else:
                    if self.__javaVectors: res.add(vector.clone().multiply(float(val)))
                    else: res += val*vector
            else: pass #print 'vector not found: [%s]' % word
        if self.__javaVectors: res = self.__word2vec.javaVector2Sparse(res)
        return res

    # requires corpus_tfidf_builder
    def __buildCorpusTfidf(self):
        if self.__tfidfBuilt: return
        builder = resolve('corpus_tfidf_builder')
        self.__ctf = builder(corpus=self.corpus, text2tokens=self.text2tokens,
                             dictionary=self.dictionary)
        self.__tfidfBuilt = True

