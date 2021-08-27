import wikivectors
from wikivectors import WordToVectorDiskMap

javaVMInitialized = False
def initVM():
    '''initialize java virtual machine'''
    global javaVMInitialized
    if not javaVMInitialized:
        try:
            wikivectors.initVM(vmargs=['-Djava.awt.headless=true',
                                    '-Dsun.arch.data.model=64', '-Dsun.cpu.endian=little'])
        except:
            wikivectors.initVM()
        javaVMInitialized = True

initVM()

def testVectors():
    vectors = WordToVectorDiskMap("/datafast/wiki_esa/wiki-esa-terms.txt",
                "/datafast/wiki_esa/wiki-esa-vectors.txt", "esa", True, True,
                "/datafast/wiki_esa/wikivectors_cache/")
    for w in ['trump', 'president']:
        vec = vectors.getWordVector(w)
        print vec

import sys, numpy as np
from scipy.sparse import lil_matrix as sparse_type
class EsaWordVectorizer():
    '''
    Reads stored esa vectors for words.
    '''

    def __init__(self, javaVectors = False):
        '''
        :param javaVectors: if True, do not convert vectors from java to scipy sparse vectors
        '''
        self.__word2vec = WordToVectorDiskMap("/datafast/wiki_esa/wiki-esa-terms.txt",
                                      "/datafast/wiki_esa/wiki-esa-vectors.txt",
                                      "esa", True, True,
                                      "/datafast/wiki_esa/wikivectors_cache/")
        self.__DIM = 100000000
        self.__cache = {}
        self.__javaVectors = javaVectors

    def __call__(self, word):
        if word in self.__cache: return self.__cache[word]
        vec = self.__word2vec.getWordVector(word)
        if self.__javaVectors: result = vec
        else:
            if vec is not None:
                result = self.javaVector2Sparse(vec)
            else: result = None
        self.__cache[word] = result
        return result

    def javaVector2Sparse(self, vec):
        result = sparse_type((1, self.__DIM), dtype=np.double)
        for e in vec.getNonZeroEntries():
            result[0, e.coordinate] = e.value
        return result

