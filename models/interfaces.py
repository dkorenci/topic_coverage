'''
Abstract interfaces for topic models and topics. 
'''
import numpy as np
import pickle, os
from utils.utils import normalize_path
from models.description import *

class TopicModel():
    '''
    Operations and data every topic models should support.
    data: 
        dictionary -  mapping of indexes to words, dict[index] returns word
        num_topics
    '''
    def topic(self, index): raise NotImplementedError

    def topic_indices(self): raise NotImplementedError

    def infer_topics(self, tokens):
        "return mapping topic_index -> topic weight for topics in the model"
        raise NotImplementedError

    def top_word_indices(self, topic_index, N=5):
        'return top N words indices for the topic'
        vec = self.topic(topic_index).vector
        #todo caching
        if len(vec) < N: N = len(vec)
        top_indices = np.argsort(vec)[::-1][:N] # get indices in sorted order, reverse, take first topN
        return top_indices

    def top_words(self, topic_index, N=5):
        words = [ self.dictionary[i] for i in self.top_word_indices(topic_index, N) ]
        return ' '.join(words)

    def topic_priors(self):
        'return map topic_index -> alpha prior'
        raise NotImplementedError

    def __getitem__(self,index):        
        return self.topic(index)

    def id(self): raise NotImplementedError

    _model_file = 'TopicModel_object'
    @staticmethod
    def load(folder):
        'load and return object that is an instance of a subclass of the topic model'
        folder = normalize_path(folder)
        model = pickle.load(open(folder + TopicModel._model_file ,'rb'))
        if os.path.exists(folder+'description.xml'):
            model.description = load_from_file(folder+'description.xml')
        else: model.description = None
        model.load(folder)
        return model

    def save(self, folder):
        '''
        Subclass must pickle itself to __model_file in the specified folder, and
        save any additional data to the folder. It must provide self.load() object
        that loads this additional data in the object.
        '''
        raise NotImplementedError

    def perplexity(self, docs): raise NotImplementedError

class Topic():
    '''
    Operations and data every topic should support
    data: model - TopicModel, vector - ndarray or compatible object
    '''       
    def __init__(self, model, vector):
        self.model = model
        self.vector = vector    
