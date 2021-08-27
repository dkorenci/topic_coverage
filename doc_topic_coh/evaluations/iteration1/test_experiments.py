from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer, scorersFromParams
from doc_topic_coh.evaluations.tools import topicMeasureAuc, \
    flattenParams as fp, joinParams as jp
from doc_topic_coh.dataset.topic_splits import iter0DevTestSplit
from pytopia.measure.topic_distance import cosine, l1, l2
from doc_topic_coh.evaluations.experiment import IdDict, IdList, \
    TopicScoringExperiment as TSE

def testWord2Vec():
    params = {'type': 'variance',
              'vectors': [ 'word2vec' ],
              'threshold': 100,
              'distance': cosine,
              'center': ['mean', 'median'],
              'exp': [0.5, 1.0, 2.0] }
    p = IdList(fp(params))
    p.id = 'word2vec_test'
    return p

def testMatrixParams():
    params = {'type': ['variance', 'avg-dist'],
              'vectors': [ 'tf-idf', 'probability' ],
              'threshold': [100, 50],
              'distance': [cosine, l2],
              'center': ['mean', 'median'],
              'exp': 1.0, }
    p = IdList(fp(params))
    p.id = 'test_matrix_params'
    return p

def testGraphParams():
    basic = {'type':'graph',
                    'vectors': ['word2vec'],
                   #'vectors': [ 'tf-idf', 'probability' ],
                   'threshold': [50, 100]}
    dcos = { 'distance': cosine,
             'algorithm': ['clustering', 'closeness'],
             'weightFilter': [[0, 0.9]],
             'weighted': [True, False],
             'center': ['mean', 'median'], }
    dl2 = { 'distance': l2, 'algorithm': 'closeness',
            'weightFilter': None, 'weighted': True,
            'center': ['mean', 'median'], }
    p = IdList(jp(fp(basic), fp(dcos)) + jp(fp(basic), fp(dl2)))
    p.id = 'test_graph_word2vec'
    return p

def testDensParams():
    basic = {'type':'density',
                    'vectors': ['word2vec'],
                   #'vectors': [ 'tf-idf', 'probability'],
                   'threshold': [50, 100],
                   'covariance': ['diag', 'spherical'],
                   'scoreMeasure': ['ll', 'bic'],
                   'dimReduce': [None, 20] }
    basic = IdList(fp(basic))
    basic.id = 'test_density_word2vec'
    return basic


def testBaselineParams():
    #p = { 'type': ['npmi', 'uci', 'umass', 'c_a', 'c_p', 'c_v', ] }
    p = { 'type': ['text_distribution', 'pairwise_word2vec', 'tfidf_coherence'] }
    p = IdList(fp(p))
    p.id = 'test_baseline'
    print p
    return p

def testAuc(params, measure=True):
    dev, test = iter0DevTestSplit()
    ltopics = dev
    scorers = scorersFromParams(params)
    for s in scorers:
        print s.id
        m = s()
        if measure:
            res = topicMeasureAuc(m, ltopics, ['theme', 'theme_noise'])
            print '%.4f' % res

from os import path
expFolder = '/datafast/doc_topic_coherence/experiments/test/'
from os import path
def testExperiment():
    dev, test = iter0DevTestSplit()
    #params = testWord2Vec()
    #params = testGraphParams()
    #params = testDensParams()
    #params = testBaselineParams()
    params = testMatrixParams()
    tse = TSE(paramSet=params, scorerBuilder=DocCoherenceScorer,
              ltopics=dev, posClass=['theme', 'theme_noise'],
              folder=expFolder, cache=True)
    tse.run()
    tse.printResults()

if __name__ == '__main__':
    #testAuc(testMatrixParams(), False)
    #testAuc(testGraphParams(), True)
    #testAuc(testDensParams(), True)
    testExperiment()
    #testAuc(testMatrixParams(), True)

