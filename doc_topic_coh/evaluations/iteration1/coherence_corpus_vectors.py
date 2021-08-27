from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer, scorersFromParams
from doc_topic_coh.evaluations.tools import topicMeasureAuc, \
    flattenParams as fp, joinParams as jp
from doc_topic_coh.dataset.topic_splits import iter0DevTestSplit
from pytopia.measure.topic_distance import cosine, l1, l2
from doc_topic_coh.evaluations.experiment import IdDict, IdList, \
    TopicScoringExperiment as TSE

def matrixParamsCorpusVectors():
    params = {'type': ['variance', 'avg-dist'],
              'vectors': [ 'tf-idf', 'probability' ],
              'threshold': [100, 50],
              'distance': [cosine, l2],
              'center': ['mean', 'median'],
              'exp': [1.0], }
    p = IdList(fp(params))
    p.id = 'matrix_params_corpus_vectors'
    return p

def graphParams():
    basic = {'type':'graph',
                   'vectors': [ 'tf-idf', 'probability' ],
                   'threshold': [50, 100]}
    dcos = { 'distance': cosine,
             'algorithm': ['clustering', 'closeness'],
             'weightFilter': [[0,0.5], [0,0.7], [0,0.8], [0,0.9]],
             'weighted': [True, False],
             'center': ['mean', 'median'], }
    dl2 = { 'distance': l2, 'algorithm': 'closeness',
            'weightFilter': None, 'weighted': True,
            'center': ['mean', 'median'], }
    p = IdList(jp(fp(basic), fp(dcos)) + jp(fp(basic), fp(dl2)))
    p.id = 'graph_params_corpus_vectors'
    return p

def densityParamsCorpusVectors():
    basic = {'type':'density',
                   'vectors': [ 'tf-idf', 'probability'],
                   'threshold': [50, 100],
                   'covariance': ['diag', 'spherical'],
                   'scoreMeasure': ['ll', 'bic', 'aic'],
                   'dimReduce': [None, 10, 20, 50, 100] }
    basic = IdList(fp(basic))
    basic.id = 'density_params_corpus_vectors'
    return basic


dev, test = iter0DevTestSplit()
expFolder = '/datafast/doc_topic_coherence/experiments/iter1_coherence/'
def experimentMatrixCorpus():
    tse = TSE(paramSet=matrixParamsCorpusVectors(), scorerBuilder=DocCoherenceScorer,
              ltopics=dev, posClass=['theme', 'theme_noise'], folder=expFolder)
    #tse.run()
    tse.printResults()
    #tse.testSignificance(0, 30, N=10000)
    #method="bootstrap"
    method="delong"
    #method = "venkatraman"
    for i in range(1,31):
        tse.testSignificance(0,i,N=8000,method=method)

def experimentGraphCorpus():
    tse = TSE(paramSet=graphParams(), scorerBuilder=DocCoherenceScorer,
              ltopics=dev, posClass=['theme', 'theme_noise'], folder=expFolder)
    tse.run()
    #tse.printResults()
    tse.significance()

def experimentDensCorpus():
    tse = TSE(paramSet=densityParamsCorpusVectors(), scorerBuilder=DocCoherenceScorer,
              ltopics=dev, posClass=['theme', 'theme_noise'], folder=expFolder)
    #tse.run()
    tse.printResults()

if __name__ == '__main__':
    #experimentMatrixCorpus()
    experimentGraphCorpus()
    #experimentDensCorpus()
