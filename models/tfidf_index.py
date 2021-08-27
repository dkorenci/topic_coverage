from gensim.models.tfidfmodel import TfidfModel
from utils.utils import normalize_path

from heapq import *
import pickle, os

class HeapItem():
    'holder of (docId, tfidf), with comparison by tfidf (ascending) operator'
    def __init__(self, docId, tfidf): self.docId, self.tfidf = docId, tfidf
    def __cmp__(self, other):
        if self.tfidf < other.tfidf : return -1
        elif self.tfidf == other.tfidf : return 0
        else : return 1

class CorpusTfidfIndex():
    def __init__(self, corpus, dictionary, tokenizer, topDocs = 200):
        '''
        :param corpus: instance of Corpus
        :param dictionary: gensim dictionary
        :param tokenizer: callable, accepts string, returns list of string (tokens)
        :param topDocs: max. number of top weight documents that will be stored for each word
        '''
        self.topDocs = topDocs
        self.numWords = len(dictionary)
        self.tokenizer = tokenizer
        self.__initIndex()
        self.tfidf = TfidfModel(id2word=dictionary, dictionary=dictionary)
        for txto in corpus:
            tfidfBow = self.tfidf[dictionary.doc2bow(tokenizer(txto.text))]
            for wordId, weight in tfidfBow:
                self.__addToIndex(wordId, txto.id, weight)
        self.__transformIndex()
        #test (on test corupus), build interface, budild

    def getTopDocs(self, word, numDocs = None):
        'get numDocs top documents by tfidf for word, as a list of (docId, tfidf) pairs'
        if isinstance(word, (str, unicode)): word = self.tfidf.id2word.token2id[word]
        if numDocs is None or numDocs > self.topDocs : numDocs = self.topDocs
        if word > self.numWords : return []
        if self.word2doc[word] is None : return []
        return [ pair for pair in self.word2doc[word][:numDocs] ]

    def __initIndex(self):
        self.word2doc = [None] * self.numWords

    def __addToIndex(self, wordId, docId, tfidf):
        '''
        add (wordId, tfidf) data to heap of such pairs for wordId,
        maintaining the max. heap size at topDocs (largest by tfidf) pairs
        '''
        item = HeapItem(docId, tfidf)
        if self.word2doc[wordId] is None:
            self.word2doc[wordId] = [ item ]
            heapify(self.word2doc[wordId])
        else:
            if len(self.word2doc[wordId]) < self.topDocs :
                heappush(self.word2doc[wordId], item)
            else: heappushpop(self.word2doc[wordId], item)

    def __transformIndex(self):
        '''
        transform heaps with word-document data to a list
        of (docId, tfidf) pairs, sorted descending by tfidf
        '''
        for wordId in range(self.numWords):
            if self.word2doc[wordId] is None: continue
            sortedItems = sorted([i for i in self.word2doc[wordId]])[::-1]
            self.word2doc[wordId] = [(item.docId, item.tfidf) for item in sortedItems]

    __gensim_file = 'gensim_tfidfmodel'
    index_file = 'tfidfmodel.pickle'
    def save(self, folder):
        # todo generalize saving and loading to folder for classes
        # that contain gensim models
        folder = normalize_path(folder)
        if not os.path.exists(folder): os.makedirs(folder)
        self.tfidf.save(folder + CorpusTfidfIndex.__gensim_file)
        self.__clear_gensim_data()
        pickle.dump(self, open(folder + CorpusTfidfIndex.index_file, 'wb'))

    def __clear_gensim_data(self):
        self.tfidf = None

    def __init_gensim_data(self): return

    @staticmethod
    def load(folder):
        folder = normalize_path(folder)
        index = pickle.load(open(folder + CorpusTfidfIndex.index_file ,'rb'))
        index.loadGensim(folder)
        return index

    def loadGensim(self, folder):
        folder = normalize_path(folder)
        self.tfidf = TfidfModel.load(folder+CorpusTfidfIndex.__gensim_file)
        self.__init_gensim_data()