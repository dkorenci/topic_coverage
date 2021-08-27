'''
Adapters from various models to abstract model.
'''

from models.interfaces import TopicModel, Topic
from models.label import modelLabel
from gensim.models import LdaModel
from gensim_mod.models.ldamodel import LdaModel as LdaModel_mod
from utils.utils import normalize_path
from preprocessing.text2tokens import RsssuckerTxt2Tokens

import pickle, os, numpy as np

class GensimLdamodel(TopicModel):
    "adapter for gensim's LdaModel class"
    
    def __init__(self, model, tag = ''):
        self.tag = tag
        if model is not None:
            if not isinstance(model, (LdaModel, LdaModel_mod)) :
                raise TypeError('model must be of type ldamodel')
            self.__init_gensim_data(model)

    def id(self):
        return modelLabel(self.model)

    def __init_gensim_data(self, model):
        'init model related data from LdaModel'
        self.model = model
        self.state = model.state
        self.dictionary = model.id2word
        self.num_topics = model.num_topics

    def __clear_gensim_data(self):
        'set all LdaModel data to none'
        self.model, self.state, self.dictionary, self.num_topics = \
            None, None, None, None

    def topic_indices(self):
        return range(self.num_topics)

    def topic(self, index):
        topic = self.state.get_lambda()[index]
        topic = topic / topic.sum() 
        return Topic(self, topic)

    def infer_topics(self, tokens):
        bow = self.dictionary.doc2bow(tokens, allow_update=False)
        result = self.model.inference([bow], collect_sstats=False)
        theta = result[0][0] #first part of the result 2-tuple is a list of vectors, take first vector
        theta /= theta.sum() # normalize to distribution
        return theta

    def name(self): return 'ldagensim'+self.tag

    __gensim_file = 'gensim_ldamodel'
    def save(self, folder):
        folder = normalize_path(folder)
        if not os.path.exists(folder): os.makedirs(folder)
        self.model.save(folder + GensimLdamodel.__gensim_file)
        self.__clear_gensim_data()
        pickle.dump(self, open(folder + TopicModel._model_file, 'wb'))

    def load(self, folder):
        folder = normalize_path(folder)
        gensim_model = LdaModel.load(folder+GensimLdamodel.__gensim_file)
        if not hasattr(self, 'text2tokens'):
            self.text2tokens = RsssuckerTxt2Tokens()
        self.__init_gensim_data(gensim_model)

    def perplexity(self, docs):
        'return perplexity for bag of words represented documents'
        bound = self.model.log_perplexity(docs)
        pplexity = np.exp2(-bound)
        return pplexity

    def topic_priors(self):
        alphas = {}
        for i in range(self.model.num_topics): alphas[i] =  self.model.alpha[i]
        return alphas