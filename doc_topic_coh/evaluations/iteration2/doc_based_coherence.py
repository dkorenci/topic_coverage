from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer, scorersFromParams
from doc_topic_coh.evaluations.tools import topicMeasureAuc, \
    flattenParams as fp, joinParams as jp
from doc_topic_coh.dataset.topic_splits import devTestSplit2
from pytopia.measure.topic_distance import cosine, l1, l2
from doc_topic_coh.evaluations.experiment import IdDict, IdList, \
    TopicScoringExperiment as TSE

def distanceParams():
    params = {'type': ['variance', 'avg-dist'],
              'vectors': None,
              'threshold': [100, 50],
              'distance': [cosine, l2],
              'center': ['mean', 'median'],
              'exp': [1.0], }
    p = IdList(fp(params))
    p.id = 'distance_params'
    return p

def graphParams():
    basic = {'type':'graph',
                   'vectors': None,
                   'threshold': [50, 100]}
    dcos = { 'distance': cosine,
             'algorithm': ['clustering', 'closeness',
                           'communicability', 'num_connected'],
             'weightFilter': [ [0,0.9], [0, 0.95] ],
             'weighted': [True, False],
             'center': ['mean', 'median'], }
    dl2 = { 'distance': [cosine, l2], 'algorithm': ['closeness', 'communicability'],
            'weightFilter': None, 'weighted': True,
            'center': ['mean', 'median'], }
    p = IdList(jp(fp(basic), fp(dcos)) + jp(fp(basic), fp(dl2)))
    p.id = 'graph_params'
    return p

def densityParams():
    basic = {'type':'density',
                   'vectors': None,
                   'threshold': [50, 100],
                   'covariance': ['diag', 'spherical'],
                   'scoreMeasure': ['ll'],
                   'dimReduce': [None, 10, 20, 50, 100] }
    basic = IdList(fp(basic))
    basic.id = 'density_params'
    return basic

def matrixParams():
    params = {
              'type': ['matrix'],
              'vectors': None,
              'threshold': [100, 50],
              'method': ['mu', 'mu0', 'mu1']
               }
    p = IdList(fp(params))
    p.id = 'matrix_params'
    return p

def assignVectors(params, vectors):
    '''
    Create new parameters by assigning 'vectors' parameters,
     for either corpus or world vectors.
    '''
    if vectors == 'world': vec = ['word2vec']
    elif vectors == 'corpus': vec = ['tf-idf', 'probability']
    else: raise Exception('invalid vectors')
    newp = IdList()
    for p in params:
        for vp in vec:
            np = p.copy()
            np['vectors'] = vp
            newp.append(np)
    newp.id = params.id + '_%s_vectors' % vectors
    return newp

dev, test = devTestSplit2()
expFolder = '/datafast/doc_topic_coherence/experiments/iter2_coherence/'
def experimentDistance(vectors):
    params = assignVectors(distanceParams(), vectors)
    print params.id
    tse = TSE(paramSet=params, scorerBuilder=DocCoherenceScorer,
              ltopics=dev, posClass=['theme', 'theme_noise'], folder=expFolder, cache=True)
    #tse.run()
    #tse.printResults()
    tse.significance(0.05)

def experimentGraph(vectors):
    params = assignVectors(graphParams(), vectors)
    print params.id
    tse = TSE(paramSet=params, scorerBuilder=DocCoherenceScorer,
              ltopics=dev, posClass=['theme', 'theme_noise'], folder=expFolder, cache=True)
    #tse.run()
    tse.printResults()
    #tse.significance(0.05)

def experimentDensity(vectors):
    params = assignVectors(densityParams(), vectors)
    print params.id
    tse = TSE(paramSet=params, scorerBuilder=DocCoherenceScorer,
              ltopics=dev, posClass=['theme', 'theme_noise'], folder=expFolder, cache=True)
    #tse.run()
    #tse.printResults()
    tse.significance(0.05)

def experimentMatrix(vectors):
    params = assignVectors(matrixParams(), vectors)
    tse = TSE(paramSet=params, scorerBuilder=DocCoherenceScorer,
              ltopics=dev, posClass=['theme', 'theme_noise'], folder=expFolder, cache=True)
    #tse.run()
    #tse.printResults()
    tse.significance(0.05)

def countParams():
    dp = len(distanceParams())*4
    gp = len(graphParams())*4
    pp = len(densityParams())*4
    mp = len(matrixParams())*4
    print 'dist %d, graph %d, dens %d, matrix %d' % (dp, gp, pp, mp)
    print 'all', dp+gp+pp+mp

if __name__ == '__main__':
    #countParams()
    #experimentDistance('corpus')
    #experimentDistance('world')
    experimentGraph('corpus')
    #experimentGraph('world')
    #experimentDensity('corpus')
    #experimentDensity('world')
    #experimentMatrix('corpus')
    #experimentMatrix('world')
