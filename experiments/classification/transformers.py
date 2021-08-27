'''
scikit-learn text to feature vector transformers
'''

from resources.resource_builder import *

import numpy as np
from corpus.factory import CorpusFactory
from sklearn.feature_extraction.text import CountVectorizer
from numpy.linalg import norm

from scipy.sparse import dok_matrix
import os, pickle #, joblib

def sparseRowToRow(m, r, sr):
    'copy row of a sparse matrix to a row of ndarray'
    for c,val in sr.items(): m[r, c[1]] = val

def vectorCacheFile(corpus):
    return object_store+('mappers/vectors/%s.pickle' % (corpus))

def loadCorpusVectors(corpus):
    print 'loading vectors from %s' % vectorCacheFile(corpus)
    return pickle.load(open(vectorCacheFile(corpus), 'rb'))

def storeCorpusVectors(instance):
    vectorCache = {}
    count = -1
    for txto in instance.corpus:
        vectorCache[txto.id] = instance.createVector(txto)
        count -= 1
        if count == 0 : break
    print vectorCacheFile(instance.corpus_id)
    pickle.dump(vectorCache, open(vectorCacheFile(instance.corpus_id), 'wb'))

class CorpusTransformer():
    '''
    transforms Text objects to feature vectors for a single corpus,
    dictionary and tf-idf data are defined for a corpus.
    '''
    @staticmethod
    def loadVectorCache():
        if hasattr(CorpusTransformer, 'vectorCache'):
            return CorpusTransformer.vectorCache
        else:
            if os.path.exists(vectorCacheFile('us_politics')):
                print 'loading vector cache'
                CorpusTransformer.vectorCache = loadCorpusVectors('us_politics')
            else:
                CorpusTransformer.vectorCache = None
            CorpusTransformer.vectorCache

    def __init__(self, corpus_id, normalize = None):
        self.corpus_id = corpus_id
        self.normalize = normalize
        self.__init_resources()

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return { 'corpus_id' : self.corpus_id, 'normalize': self.normalize }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.__init_resources()
        return self

    def __str__(self):
        return 'CorpusTransformer: corpus_id %s, normalize %s' % (self.corpus, self.normalize)

    def __init_resources(self):
        self.txt2tok = RsssuckerTxt2Tokens()
        self.vectorCache = CorpusTransformer.loadVectorCache()
        self.corpus = CorpusFactory.getCorpus(self.corpus_id)
        # todo: make dict and tfidf objects per-corpus singletons
        self.dict = loadDictionary(self.corpus_id)
        self.N = len(self.dict)
        self.tfidf = loadTfidfIndex(self.corpus_id)

    def __getstate__(self):
        return self.corpus_id, self.txt2tok, self.normalize

    def __setstate__(self, state):
        self.corpus_id, self.txt2tok, self.normalize  = state
        self.__init_resources()

    def fit_transform(self, texts, y=None):
        return self.transform(texts)

    def transform(self, texts, y=None):
        #print 'transforming texts of length %d' % len(texts)
        D = len(texts)
        m = np.zeros((D, self.N))
        # construct numpy array
        r = 0
        for txto in texts:
            if self.vectorCache is not None and (txto.id in self.vectorCache) :
                sparseRowToRow(m, r, self.vectorCache[txto.id])
            else:
                sr = self.createVector(txto)
                sparseRowToRow(m, r, sr)
                #self.vectorCache[txto.id] = sr
            if self.normalize == 'unit-vector':
                m[r] /= norm(m[r])
            r += 1
        return m

    def createVector(self, txto):
        #v = np.zeros(self.N)
        v = dok_matrix((1,self.N), dtype=float)
        for wi, freq in self.dict.doc2bow(self.txt2tok(txto.text)):
            v[0, wi] = (1+np.log2(freq))*self.tfidf.tfidf.idfs[wi]
        return v

