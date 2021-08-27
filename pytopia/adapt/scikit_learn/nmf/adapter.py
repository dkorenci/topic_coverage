from pytopia.topic_model.TopicModel import TopicModel, Topic
from sklearn.externals import joblib
from pytopia.context.ContextResolver import resolve, resolveIds
from pytopia.tools.IdComposer import IdComposer
from pytopia.tools.logging import resbuild_logger

from sklearn.decomposition import NMF
from scipy.sparse import dok_matrix
import numpy as np

import os, pickle
from os import path

@resbuild_logger
class SklearnNmfBuilder():

    def __init__(self): pass

    def __call__(self, *args, **kwargs):
        nmfa = SklearnNmfTmAdapter(*args, **kwargs)
        nmfa.build()
        return nmfa

    def resourceId(self, *args, **kwargs):
        return SklearnNmfTmAdapter(*args, **kwargs).id

class SklearnNmfTmAdapter(TopicModel, IdComposer):
    '''
    Adapts sklearn NMF model to pytopia TopicModel
    '''

    def __init__(self, corpus, dictionary, text2tokens, T,
                 preproc='tf-idf', rndSeed=1, **options):
        '''
        :param corpus: pytopia corpus
        :param dictionary: pytopia dictionary
        :param txt2tokens: pytopia text2tokens
        :param T: number of topics, ie NMF components
        :param preprocess: 'tf-idf' for using tf-idf representation of documents,
                            or 'word-prob', for word probabilities
        :param rndSeed: random_state for NMF builder
        :param options: other options for nmf builder
        :return:
        '''
        self.corpus, self.dictionary, self.text2tokens = \
            resolveIds(corpus, dictionary, text2tokens)
        self.T, self.preproc, self.rndSeed = T, preproc, rndSeed
        atts = ['corpus', 'dictionary', 'text2tokens', 'T', 'preproc', 'rndSeed']
        # add sklearn NMF options as object's attributes
        for k, v in options.iteritems():
            self.__dict__[k] = v
            atts.append(k)
        # todo solve by automatically excluding class attributes from id composition
        IdComposer.__init__(self, attributes=atts)

    ### TopicModel interface methods

    def topicIds(self): return range(self.T)

    def numTopics(self): return self.T

    def topic(self, topicId):
        return Topic(self, topicId, self.topicVector(topicId))

    def topicVector(self, topicId):
        return self.__nmf.components_[topicId]

    def inferTopics(self, txt, batch=True, format='bow'):
        if format != 'bow': return None # todo: support other formats
        texts = [txt] if not batch else txt
        dict = resolve(self.dictionary)
        from pytopia.corpus.tools import bowCorpus2Matrix
        matrix = bowCorpus2Matrix(texts, dict)
        res = self.__nmf.transform(matrix)
        return res

    def corpusTopicVectors(self):
        return self.__w

    # requires corpus_index_builder
    def build(self):
        '''
        Builds the NMF model from the init data.
        :return:
        '''
        if self.preproc == 'word-prob': matrix = self.__buildWordProbMatrix()
        elif self.preproc == 'tf-idf': matrix = self.__buildTfidfMatrix()
        else: raise Exception('unknown preprocessing: %s' % self.preproc)
        nmf = NMF(n_components=self.T, random_state=self.rndSeed, solver='pg', init='nndsvd')
        w = nmf.fit_transform(matrix)
        self.__nmf = nmf
        self.__w = w

    def __buildTfidfMatrix(self):
        builder = resolve('corpus_tfidf_builder')
        tfidf = builder(corpus=self.corpus, dictionary=self.dictionary,
                        text2tokens=self.text2tokens)
        return tfidf.corpusMatrix()

    def __buildWordProbMatrix(self):
        bowBuilder = resolve('bow_corpus_builder')
        bowCorpus = bowBuilder(corpus=self.corpus, text2tokens=self.text2tokens,
                               dictionary=self.dictionary)
        matrix = bowCorpus.corpusMatrix(dtype=np.float64)
        # switch word counts to word probabilities, ie normalize rows by number of words per row
        # TODO maybe move to BowCorpus
        rowSums = matrix.sum(1)
        for i in xrange(matrix.shape[0]):
            numWords = rowSums[i, 0]
            if numWords > 0:
                matrix[i] /= numWords
        return matrix

    nmfModelFile = 'sklearnNmf.pkl'
    corpusTopicsFile = 'corpusTopicWeights.pkl'
    def save(self, folder):
        if not path.exists(folder): os.makedirs(folder)
        # todo: generic pytopia-level handling of large matrix attributes
        joblib.dump(self.__nmf, path.join(folder, self.nmfModelFile))
        joblib.dump(self.__w, path.join(folder, self.corpusTopicsFile))
        nmf = self.__nmf; self.__nmf = None
        w = self.__w; self.__w = None
        from pytopia.resource.loadSave import objectSaveFile
        pickle.dump(self, open(path.join(folder, objectSaveFile), 'wb'))
        self.__nmf = nmf
        self.__w = w

    def load(self, folder):
        self.__nmf = joblib.load(path.join(folder, self.nmfModelFile))
        self.__w = joblib.load(path.join(folder, self.corpusTopicsFile))
        TopicModel.load(self, folder)