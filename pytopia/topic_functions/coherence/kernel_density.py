from pytopia.tools.IdComposer import IdComposer
from pytopia.resource.tools import tfIdfMatrix

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import cross_val_score

import numpy as np

class KernelDensityCoherence(IdComposer):
    '''
    Calculates coherence as probability of the tf-idf vectors of
     top documetns with regard to fitted kernel density model.
    '''

    def __init__(self, seed=8932183):
        self.seed = seed
        IdComposer.__init__(self)

    # requires corpus_topic_index_builder
    # requires corpus_tfidf_builder
    def __call__(self, topic):
        '''
        :param topic: (modelId, topicId)
        :return:
        '''
        m = tfIdfMatrix(topic, 100)
        gridParams = {'bandwidth': np.logspace(-1, 1, 10)}
        fitCV = 5
        gridSearch = GridSearchCV(KernelDensity(), gridParams, cv=fitCV, n_jobs=3)
        gridSearch.fit(m)
        estimator = gridSearch.best_estimator_
        scoreCV = 5
        scores = cross_val_score(estimator, m, cv=scoreCV, n_jobs=3)
        r = scores.sum() / scoreCV
        print scores, r
        return r

