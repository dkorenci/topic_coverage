import os
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


# files for storing corpus resources
def dictFile(corpusId): return object_store + corpusId + '_dict'
def bowCorpusFile(corpusId): return object_store + corpusId + '_bow_corpus'

def getCorpus(corpus):
    '''
    :param corpus: instance of Corpus or corpus Id
    :return: param if its a Corpus or corpus with corresponding id
    '''
    if isinstance(corpus, Corpus): return corpus
    else: return CorpusFactory.getCorpus(corpus)

def buildDictionaryAndBow(corpus, text2tokens = RsssuckerTxt2Tokens(), verbose = True):
    '''
    for the Text's in a corpus, and specified tokenization,
    build dictionary (token to int id mapping) and corpus containing
    bag-of-word representations (lists of (token-id, token-count) pairs) of documents
    : param
    '''
    corpus = getCorpus(corpus)
    if verbose :
        print 'corpus name: %s' % corpus.corpusId()
        print 'corpus size: %d' % len(corpus)
    dictionary = buildDictionary(corpus, text2tokens, verbose)
    bowCorpus = buildBowCorpus(corpus, dictionary, text2tokens, verbose)
    return dictionary, bowCorpus

def buildDictionary(corpus, text2tokens = RsssuckerTxt2Tokens(), verbose = True, save=True):
    'build and store gensim dictionary'
    corpus = getCorpus(corpus)
    if verbose: print 'building dictionary for %s' % corpus.corpusId()
    dictionary = Dictionary(documents=None)
    tokstream = Text2TokensStream(corpus, text2tokens)
    t = clock()
    for doc in tokstream:
        _ = dictionary.doc2bow(doc, allow_update=True)
    dictionary.filter_extremes(no_below=5, no_above=0.1, keep_n=1000000)
    dictionary.compactify()
    if verbose:
        print 'dictionary size: %d' % len(dictionary)
        print 'dictionary build time: ' + str(clock()-t)
    if save: dictionary.save(dictFile(corpus.corpusId()))
    return dictionary

def loadDictionary(corpusId, create = True):
    if not os.path.exists(dictFile(corpusId)) and create:
        return buildDictionary(corpusId)
    else:
        return Dictionary.load(dictFile(corpusId))

def buildBowCorpus(corpus, dictionary, text2tokens = RsssuckerTxt2Tokens(), verbose=True, save=True):
    corpus = getCorpus(corpus)
    if verbose: print 'building bow corpus for %s' % corpus.corpusId()
    t = clock()
    bowCorpus = bowstreamToArray(BowStream(Text2TokensStream(corpus, text2tokens), dictionary))
    if verbose: print 'bow corpus build time: ' + str(clock()-t)
    if save: np.save(bowCorpusFile(corpus.corpusId()), bowCorpus)
    return bowCorpus

def loadBowCorpus(corpusId, create = True):
    file = bowCorpusFile(corpusId)+'.npy'
    if not os.path.exists(file) and create:
        print 'bow corpus does not exists for %s , creating it' % corpusId
        return buildBowCorpus(corpusId, loadDictionary(corpusId))
    else:
        return np.load(file) #, mmap_mode='r')

def modelFolder(modelName): return models_folder+modelName
def buildModel(corpusId, modelOptions, save = True, modelName = None,
               text2tokens = RsssuckerTxt2Tokens(), verbose = True):
    dict = loadDictionary(corpusId)
    bowstream = loadBowCorpus(corpusId)
    gensimModel = modelOptions.getModel(dict)
    t = clock()
    gensimModel.update(bowstream)
    if verbose: print 'model build time: ' + str(clock()-t)
    model = GensimLdamodel(gensimModel)
    if text2tokens is not None:
        model.text2tokens = text2tokens
    if save:
        if modelName is None:
            modelName = corpusId + '_' + modelLabel(modelOptions)
        folder = modelFolder(modelName)
        if not os.path.exists(folder): os.makedirs(folder)
        model.save(folder)

def wrapGensimLdaModel(file, label):
    model = LdaModel.load(file)
    wrapper = GensimLdamodel(model)
    wrapper.save(models_folder+modelLabel(model)+str(label))

def buildTfidfIndex(corpusId, tokenizer = RsssuckerTxt2Tokens()):
    print 'building tfidf index for corpus %s' % corpusId
    corpus = getCorpus(corpusId)
    dict = loadDictionary(corpusId)
    tfidfIndex = CorpusTfidfIndex(corpus, dict, tokenizer)
    tfidfIndex.save(tfidfindex_folder+'%s_index'%corpusId)

def loadTfidfIndex(corpusId):
    corpusIndexFolder = tfidfindex_folder+'%s_index'%corpusId
    if not os.path.exists(corpusIndexFolder): buildTfidfIndex(corpusIndexFolder)
    return CorpusTfidfIndex.load(corpusIndexFolder)