'''
Functionality for loading and generating training data for model unit tests.
'''

import numpy as np
from os import path

def croelectNews(firstN = None):
    '''Load dataset derived from Croatian news articles.'''
    fname = path.join(path.dirname(__name__), 'data', 'croelect_news.npy')
    corpus = np.load(fname)
    if firstN: corpus = corpus[:firstN]
    return corpus

def singleton():
    '''corpus with one document and one word'''
    return [[(0,1)]]

def manySingletons():
    '''corpus with many documents, one word each'''
    return [[(0,1)] for i in range(1000)]

def testCorpus1():
    '''corpus with many documents, all same number of distinct words'''
    return [[(j,1) for j in range(200)] for i in range(1000)]