import os, pickle
from os import path
from time import clock

from gensim.corpora import Dictionary
import numpy as np

from pymedialab_settings.settings import *
from preprocessing.text2tokens import *
from corpus.textstream.utils import Text2TokensStream
from corpus.textstream.utils import BowStream, bowstreamToArray
from corpus.corpus import Corpus
from corpus.factory import CorpusFactory
from gensim_mod.models.ldamodel import LdaModel
from models.label import modelLabel
from models.adapters import GensimLdamodel
from models.tfidf_index import CorpusTfidfIndex
from models.topic_index import CorpusTopicsIndex
from models.interfaces import TopicModel


class ResourceBuilder:
    #todo include all the creation resources (text2tokens, ...) ids into the file name

    def __init__(self, corpusFactory, objectStore, modelsFolder, text2tokens,
                 dictCorpusId = None, verbose = True):
        '''
        :param corpusFactory: a CorpusFactory compatible object
        :param objectStore: folder for storing serialized resource objects
        :param modelsFolder: folder for storing gensim models
        :param text2tokens: text to list of tokens converter
        :param dictCorpusId: if not None, dictionary for this corpus will be
                used instead of the one for corpus given as operation parameter.
        '''
        self.corpusFactory = corpusFactory
        self.objectStore = objectStore
        self.text2tokens = text2tokens
        self.verbose = verbose
        self.modelsFolder = modelsFolder
        self.dictCorpusId = dictCorpusId

    # file names for storing resources
    def dictFile(self, corpusId):
        return path.join(self.objectStore, corpusId+'_dict')
    def bowCorpusFile(self, corpusId):
        return path.join(self.objectStore, corpusId+'_bow_corpus.npy')
    def modelFolder(self, id, modelOptions = None):
        if modelOptions: # include both id and options in folder name
            return path.join(self.modelsFolder, self.modelName(id, modelOptions))
        else: return path.join(self.modelsFolder, id)
    def modelName(self, corpusId, modelOptions):
        '''Create model folder name based on modelId and construct options'''
        return corpusId + '_' + modelLabel(modelOptions)
    def tfidfFile(self, corpusId): return self.objectStore + corpusId + '_tfidfIndex'

    def getCorpus(self, corpus):
        '''
        :param corpus: instance of Corpus or corpus Id
        :return: param if its a Corpus or corpus with corresponding id
        '''
        if isinstance(corpus, Corpus): return corpus
        else: return self.corpusFactory.getCorpus(corpus)

    def buildDictionaryAndBow(self, corpus):
        '''
        for the Text's in a corpus, and specified tokenization,
        build dictionary (token to int id mapping) and corpus containing
        bag-of-word representations (lists of (token-id, token-count) pairs) of documents
        : param
        '''
        corpus = self.getCorpus(corpus)
        if self.verbose :
            print 'corpus name: %s' % corpus.corpusId()
            print 'corpus size: %d' % len(corpus)
        dictionary = self.buildDictionary(corpus, self.text2tokens)
        bowCorpus = self.buildBowCorpus(corpus, dictionary, self.text2tokens)
        return dictionary, bowCorpus

    def buildDictionary(self, corpus, save=True):
        'build and store gensim dictionary'
        corpus = self.getCorpus(corpus)
        if self.verbose: print 'building dictionary for %s' % corpus.corpusId()
        dictionary = Dictionary(documents=None)
        tokstream = Text2TokensStream(corpus, self.text2tokens)
        t = clock()
        for doc in tokstream: dictionary.doc2bow(doc, allow_update=True)
        dictionary.filter_extremes(no_below=5, no_above=0.1, keep_n=1000000)
        dictionary.compactify()
        if self.verbose:
            print 'dictionary size: %d' % len(dictionary)
            print 'dictionary build time: ' + str(clock()-t)
        if save: dictionary.save(self.dictFile(corpus.corpusId()))
        return dictionary

    def loadDictionary(self, corpusId, create = True):
        if self.dictCorpusId: corpusId = self.dictCorpusId
        if not os.path.exists(self.dictFile(corpusId)):
            if create: return self.buildDictionary(corpusId)
            else: return None
        else:
            return Dictionary.load(self.dictFile(corpusId))

    def buildBowCorpus(self, corpus, dictionary, save=True):
        corpus = self.getCorpus(corpus)
        if self.verbose: print 'building bow corpus for %s' % corpus.corpusId()
        t = clock()
        bowCorpus = bowstreamToArray(
            BowStream(Text2TokensStream(corpus, self.text2tokens), dictionary))
        if self.verbose: print 'bow corpus build time: ' + str(clock()-t)
        if save: np.save(self.bowCorpusFile(corpus.corpusId()), bowCorpus)
        return bowCorpus

    def loadBowCorpus(self, corpusId, create = True):
        file = self.bowCorpusFile(corpusId)
        if not os.path.exists(file):
            if create:
                print 'bow corpus does not exists for %s , creating it' % corpusId
                return self.buildBowCorpus(corpusId, self.loadDictionary(corpusId))
            else: return None
        else:
            return np.load(file)

    def buildModel(self, corpusId, modelOptions, modelName = None,
                   overwrite = True):
        '''
        Build gensim lda model and save it to disk.
        :param modelName: if not none this will be model folder name
                            instead of the one constructed from options
        :param overwrite: if False and the model folder exists, no building and saving
        :return: full path of the model folder
        '''
        if modelName: folder = self.modelFolder(modelName)
        else: folder = self.modelFolder(corpusId, modelOptions)
        if os.path.exists(folder) and not overwrite: return folder
        dictionary = self.loadDictionary(corpusId)
        bowstream = self.loadBowCorpus(corpusId)
        gensimModel = modelOptions.getModel(dictionary)
        t = clock()
        if self.verbose: print 'starting model build'
        gensimModel.update(bowstream)
        if self.verbose: print 'model build time: ' + str(clock()-t)
        model = GensimLdamodel(gensimModel)
        model.corpusId = corpusId
        if self.text2tokens is not None:
            model.text2tokens = self.text2tokens
        if not os.path.exists(folder): os.makedirs(folder)
        model.save(folder)
        return folder

    def buildTopicIndex(self, modelfolder, corpusId, overwrite=True):
        '''
        Build and save topic index for specified model and corpus
        :param overwrite: if False and index exists, no building and saving
        :return: index if it was build, None otherwise
        '''
        if not path.exists(modelfolder): return None
        model = TopicModel.load(modelfolder)
        corpus = self.getCorpus(corpusId)
        indexFile = path.join(modelfolder, CorpusTopicsIndex.TOPIC_INDEX_FILE)
        if path.exists(indexFile) and not overwrite: return None
        index = CorpusTopicsIndex(model, corpus, model.text2tokens)
        pickle.dump(index, open(indexFile, 'wb'))
        return index

    def buildTfidfIndex(self, corpusId):
        print 'building tfidf index for corpus %s' % corpusId
        corpus = self.getCorpus(corpusId)
        dictionary = self.loadDictionary(corpusId)
        tfidfIndex = CorpusTfidfIndex(corpus, dictionary, self.text2tokens)
        tfidfIndex.save(self.tfidfFile(corpusId))

    def loadTfidfIndex(self, corpusId, create = True):
        if not os.path.exists(self.tfidfFile(corpusId)):
            if create: self.buildTfidfIndex(corpusId)
            else: return None
        return CorpusTfidfIndex.load(self.tfidfFile(corpusId))