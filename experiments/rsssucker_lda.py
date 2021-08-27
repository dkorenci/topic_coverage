import pickle

import numpy as np
from gensim.corpora import Dictionary

from preprocessing.text2tokens import *
from corpus.textstream.utils import Text2TokensStream
from corpus.textstream.utils import BowStream, bowstreamToArray
from models.adapters import GensimLdamodel
from models.topic_index import CorpusTopicsIndex
import pymedialab_settings.settings as settings
from pymedialab_settings.settings import object_store
from corpus.rsssucker import *
from corpus.filter import FilteredCorpus
from models.interfaces import TopicModel
#from gensim.models import LdaModel
from gensim_mod.models.ldamodel import LdaModel

# "rsssucker_topus1_05022015"
def createRssSuckerDictionary_old():
    dict = 'rsssucker_topus1_27022015_dict'
    #data = '/datafast/rsssucker_data/topus_27022015'
    # txtstream = FileStream(data)
    data = 'rsssucker_topus1_27022015'
    txtstream = RsssuckerCorpus(data)
    tokenizer = StemmerTokenizer()
    tokstream = Text2TokensStream(txtstream, tokenizer)
    dictionary = Dictionary(tokstream, prune_at=None)
    dictionary.save(fname = object_store+dict)
    return dictionary

def loadRssSuckerDictionary_old():
    dict = 'rsssucker_topus1_27022015_dict'
    return Dictionary.load(fname = object_store+dict)

def rsssuckerTokenStream(database):
    corpus = FilteredCorpus(RsssuckerCorpus(database), RsssuckerFilter())
    tokstream = Text2TokensStream(corpus, RsssuckerTxt2Tokens())
    return tokstream

def saveRsssuckerTokenStream(database):
    tokstream = rsssuckerTokenStream(database)
    tokstrlist = [ tokens for tokens in tokstream ]
    pickle.dump(tokstrlist, open(object_store+'tokstream_'+database, 'wb'))

def createRssSuckerDictionary(database):
    dict = Dictionary(documents=None)
    tokstream = rsssuckerTokenStream(database)
    for doc in tokstream:
        _ = dict.doc2bow(doc, allow_update=True)
    dict.filter_extremes(no_below=5, no_above=0.1, keep_n=1000000)
    dict.compactify()
    print len(dict)
    dict.save(object_store+'dict_'+database)

def loadRssSuckerDictionary(database):
    return Dictionary.load(object_store+'dict_'+database)

def trainRsssuckerModel(database):
    dict = loadRssSuckerDictionary(database)
    tokenstream = rsssuckerTokenStream(database)
    bowstream = loadBowStream_numpy(database) #BowStream(tokenstream, dict)
    #gmodel = LdaModel(corpus=bowstream, id2word = dict, num_topics = 100, alpha='auto')
    print 'training model'
    gmodel = LdaModel(corpus=bowstream, id2word = dict, num_topics = 200,
                      alpha='auto', alpha_init=0.003, passes=1)
    model = GensimLdamodel(gmodel)
    model.text2tokens = tokenstream.text2tokens
    model.save(object_store+'models/'+database+'_newpreproc_T200alphaauto003')

def loadModel(model):
    return TopicModel.load(object_store+'models/'+model)

def attachTxt2Tokens(modelName, text2tokens = RsssuckerTxt2Tokens()):
    model = TopicModel.load(object_store+'models/'+modelName)
    model.text2tokens = text2tokens
    model.save(object_store+'models/'+modelName+"_attachTxt2Tokens")

def persistBowStream_pickle(database):
    dict = loadRssSuckerDictionary(database)
    bowstream = BowStream(rsssuckerTokenStream(database), dict)
    bows = [ bow for bow in bowstream ]
    pickle.dump(bows, file(object_store+'bows_'+database, 'wb'))

def loadBowStream_pickle(database):
    return pickle.load(file(object_store+'bows_'+database, 'rb'))

def persistBowStream_numpy(database):
    dict = loadRssSuckerDictionary(database)
    bowstream = bowstreamToArray(BowStream(rsssuckerTokenStream(database), dict))
    #bowstream = bowstreamToArray(loadBowStream_pickle(database))
    np.save(object_store+'bowsnp2_'+database, bowstream)

def loadBowStream_numpy(database):
    return np.load(object_store+'bowsnp2_'+database+'.npy')#, mmap_mode='r')

def trainModel():
    dict_file = 'rsssucker_topus1_27022015_dict'
    dictionary = Dictionary.load(fname= object_store+dict_file)
    data = 'rsssucker_topus1_27022015'    
    txtstream = RsssuckerCorpus(data)
    tokenizer = StemmerTokenizer()
    tokstream = Text2TokensStream(txtstream, tokenizer)
    bowstream = BowStream(tokstream, dictionary)
    model_name = 'rsssucker_topus1_27022015_model'    
    model = LdaModel(corpus=bowstream, id2word = dictionary, num_topics = 100, alpha='auto')
    model.save(object_store+model_name)
    
def loadModel_old():
    model_name = 'rsssucker_topus1_27022015_model'    
    return LdaModel.load(object_store + model_name)

def inferDocument():
    model = GensimLdamodel(loadModel_old())
    corpus = RsssuckerCorpus('rsssucker_topus1_27022015')
    tokenizer = StemmerTokenizer()
    txt = corpus.getText(99)
    return model.infer_topics(tokenizer(txt.text))

def buildTopicIndex():
    model = GensimLdamodel(loadModel_old())
    corpus = RsssuckerCorpus('rsssucker_topus1_27022015')
    tokenizer = StemmerTokenizer()
    index = CorpusTopicsIndex(model, corpus, tokenizer)
    f = open(object_store+"topic_index", 'wb')
    pickle.dump(index, f); f.close()

def loadTopicIndex():
    f = open(object_store+"topic_index", 'rb')
    index = pickle.load(f); f.close()
    return index

def testSaveModel():
    model = GensimLdamodel(loadModel_old(), 'test_model')
    model.save(settings.models_folder+'test_model/')

def testLoadModel():
    model = GensimLdamodel.load(settings.models_folder+'test_model/')
    return model

