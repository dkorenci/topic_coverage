from PngImagePlugin import _idat
from corpus.factory import CorpusFactory
from resources.resource_builder import *
from gensim_mod.models.hdpmodel import HdpModel as HdpModelMod

def trainHdp():
    corpus = 'us_politics'
    bow = loadBowCorpus(corpus)
    dict = loadDictionary(corpus)
    # 21:08
    model = HdpModelMod(corpus=bow, id2word=dict, K=10, T=200)
    model.save('/datafast/pymedialab/models/hdp/hdp_model1')
    numTopics = len(model.m_lambda)
    print 'numTopics: %d' % numTopics
    model.show_topics(numTopics, 20)
