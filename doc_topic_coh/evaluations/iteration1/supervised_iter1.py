from doc_topic_coh.evaluations.tools import labelsMatch
from pytopia.topic_functions.coherence.doc_matrix_coh_factory import \
    avg_dist_coherence as avgcoh, variance_coherence as varcoh
from doc_topic_coh.dataset.topic_splits import iter0DevTestSplit, topicLabelStats
from doc_topic_coh.dataset.topic_labels import labelAllTopics, labelingStandard
from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer as SB
from pytopia.measure.topic_distance import cosine, l2
from doc_topic_coh.evaluations.tools import topicMeasureAuc, \
    flattenParams as fp

from doc_topic_coh.evaluations.supervised import *

import numpy as np

dev, test = iter0DevTestSplit()
expFolder = '/datafast/doc_topic_coherence/experiments/iter1_coherence/'
cache = '/datafast/doc_topic_coherence/experiments/iter1_coherence/function_cache'

# instantiate scorer form params
IS = lambda p : SB(cache=cache, **p)

############# baselines
blinePalmetto = { 'type': ['npmi', 'uci', 'umass', 'c_a', 'c_p', 'c_v', ] }
blinePalmetto = map(IS, fp(blinePalmetto))
blineNew = {'type': ['text_distribution', 'pairwise_word2vec', 'tfidf_coherence']}
blineNew = map(IS, fp(blineNew))

blineAll = blinePalmetto + blineNew

## corpus vectors
bestMatrixC = [
    { 'type': 'avg-dist', 'distance': cosine, 'center': 'median',
      'vectors': 'probability', 'exp': 1.0, 'threshold': 100, },
    {'type': 'variance', 'distance': l2, 'center': 'mean',
     'vectors': 'probability', 'exp': 2.0, 'threshold': 100, }
             ]
bestMatrixC = map(IS, bestMatrixC)

bestGraphC = [
    {'type': 'graph', 'distance':cosine, 'weighted': False, 'center': 'median',
     'algorithm': 'closeness', 'vectors': 'probability', 'threshold': 100,
     'weightFilter': [0, 0.9]},
    {'type': 'graph', 'distance': cosine, 'weighted': False, 'center': 'median',
     'algorithm': 'closeness', 'vectors': 'tf-idf', 'threshold': 100,
     'weightFilter': [0, 0.9]},
    { 'type': 'graph', 'distance': cosine, 'weighted': True, 'center': 'median',
     'algorithm': 'closeness', 'vectors': 'probability', 'threshold': 100,
     'weightFilter': [0, 0.9]}
]
bestGraphC = map(IS, bestGraphC)

bestDensityC = [
    {'type': 'density', 'scoreMeasure': 'aic', 'vectors': 'probability', 'covariance': 'spherical',
     'dimReduce': 20, 'threshold': 100 },
       {'type': 'density', 'scoreMeasure': 'aic', 'vectors': 'probability', 'covariance': 'spherical',
     'dimReduce': 50, 'threshold': 100 },
          {'type': 'density', 'scoreMeasure': 'aic', 'vectors': 'tf-idf', 'covariance': 'spherical',
     'dimReduce': 50, 'threshold': 100 }
]
bestDensityC = map(IS, bestDensityC)

bestCorpus = bestMatrixC+bestGraphC+bestDensityC
bestAlgo = [bestMatrixC[0], bestGraphC[0], bestDensityC[0]]

# word2vec vectors
bestMatrixW2V = [
    {'type': 'avg-dist', 'distance': cosine, 'center': 'median', 'vectors': 'word2vec', 'exp': 1.0,
     'threshold': 100},
       { 'type': 'avg-dist', 'distance': cosine, 'center': 'mean', 'vectors': 'word2vec', 'exp': 1.0,
     'threshold': 100}
]
bestMatrixW2V = map(IS, bestMatrixW2V)
bestGraphW2V = [
    {'type': 'graph', 'distance': cosine, 'weighted': True, 'center': 'mean', 'algorithm': 'closeness',
     'vectors': 'word2vec', 'threshold': 100, 'weightFilter': [0, 0.5]},
       {'type': 'graph', 'distance': cosine, 'weighted': True, 'center': 'mean', 'algorithm': 'closeness',
     'vectors': 'word2vec', 'threshold': 100, 'weightFilter': [0, 0.9]}
]
bestGraphW2V = map(IS, bestGraphW2V)
bestDensWord2V = [
    {'type': 'density', 'scoreMeasure': 'aic', 'vectors': 'word2vec',
     'covariance': 'diag', 'dimReduce': 10, 'threshold': 50},
    {'type': 'density', 'scoreMeasure': 'aic', 'vectors': 'word2vec',
     'covariance': 'diag', 'dimReduce': 20, 'threshold': 50}
]
bestDensWord2V = map(IS, bestDensWord2V)
bestAlgoW2V = [bestMatrixW2V[0], bestGraphW2V[0], bestDensWord2V[0]]

def sup(label, scorers, posClass = ['theme', 'theme_noise'], oneByOne=True, all=True):
    print label
    # one by one
    if oneByOne:
        for sc in scorers:
            print sc.id
            runNestedCV([sc()], test, posClass, logistic())
            runNestedCV([sc()], test, posClass, svm())
    # all together
    if all:
        print 'ALL SCORERS'
        scorers = [sc() for sc in scorers]
        #runNestedCV(scorers, test, posClass, logistic())
        #runNestedCV(scorers, test, posClass, svm())
        runNestedCV(scorers, test, posClass, randomForest())
        runNestedCV(scorers, test, posClass, knn())

if __name__ == '__main__':
    #supBestAlgo()
    #supBaseline()
    #sup('baseline_new', blineNew)
    #sup('baseline_palmetto', blinePalmetto)
    #sup('baseline_all', blineAll)
    #sup('best_algo_v2w', bestAlgoW2V)
    sup('best_algo_all', bestAlgo+bestAlgoW2V, oneByOne=False)
    #sup('all', bestAlgo + bestAlgoW2V + blineAll, oneByOne=False)