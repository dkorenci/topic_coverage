from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer, scorersFromParams
from doc_topic_coh.evaluations.tools import topicMeasureAuc, \
    flattenParams as fp, joinParams as jp
from doc_topic_coh.dataset.topic_splits import devTestSplit2, iter0DevTestSplit
from pytopia.measure.topic_distance import cosine, l1, l2, l2squared, lInf, canberra
from doc_topic_coh.evaluations.experiment import IdDict, IdList, \
    TopicScoringExperiment as TSE

docSelectParams = { 'threshold': [10, 25, 50, 100] }
docSelectParams = fp(docSelectParams)

distances = [cosine, l1, l2, l2squared, lInf, canberra]
newDistances = [l2squared, l1, lInf, canberra]

def testDistanceParams():
    params = {'type': ['variance', 'avg-dist'],
              'vectors': None,
              'threshold': [50],
              'distance': [cosine, l2, l1],
              'center': ['mean', 'median'],
              'exp': [1.0], }
    p = IdList(fp(params))
    p.id = 'distance_params'
    return p

def testGraphParams():
    basic = {'type':'graph', 'vectors': ['model'],
             'threshold': [50]}
    threshParams = { 'distance': l2,
             'algorithm': ['communicability', 'closeness'],
             'weightFilter': [[0, 0.39597], [0, 0.45666], [0, 0.51563], [0, 0.62909], ],
             'weighted': [False],
             'center': ['median'], }
    # Graph building without thresholding
    dl2 = { 'distance': [cosine, l2, l1], 'algorithm': ['closeness', 'communicability',
                                                 'clustering', 'mst'],
            'weightFilter': None, 'weighted': True,
            'center': ['median'], }
    #p = jp(fp(basic), fp(threshParams)) + jp(fp(basic), fp(dl2))
    p = jp(fp(basic), fp(threshParams)) #+ jp(fp(basic), fp(dl2))
    p = IdList(p)
    p.id = 'graph_params'
    return p

thresholds = {
    ('tf-idf','cosine'): [0.92056, 0.94344, 0.95661, 0.97246, 0.98525, 0.99478, ] ,
    ('tf-idf','l2'): [1.35688, 1.37364, 1.38319, 1.39460, 1.40374, 1.41052, ] ,
    ('tf-idf','l1'): [9.52592, 10.83310, 12.14663, 14.50215, 17.43981, 20.59608, ],
    ('probability','cosine'): [0.87706, 0.92783, 0.95097, 0.97251, 0.98635, 0.99536, ],
    ('probability','l2'): [0.12387, 0.13656, 0.14878, 0.17092, 0.20064, 0.23936, ],
    ('probability','l1'): [1.84746, 1.88496, 1.90951, 1.94139, 1.96886, 1.98925, ],
    ('word2vec','cosine'): [0.10344, 0.12718, 0.15091, 0.19408, 0.24799, 0.30865, ],
    ('word2vec','l2'): [58.13322, 78.54189, 101.04961, 146.47849, 225.47549, 334.48654, ],
    ('word2vec','l1'): [804.65845, 1087.80811, 1398.32141, 2030.68066, 3133.24609, 4659.37695, ],
    ('glove', 'cosine'): [0.05979, 0.07508, 0.09090, 0.12111, 0.16095, 0.20816, ],
    ('glove', 'l2'): [158.14783, 216.73793, 282.91806, 424.81396, 671.20081, 1034.27319, ],
    ('glove', 'l1'): [2015.83020, 2680.55078, 3449.74219, 4973.33984, 7456.08691, 10653.34375, ],
}
def prodGraphParams(vectors, distance):
    basic = {'type':'graph', 'vectors': vectors, 'distance': distance}
    # Graph building with thresholding
    th = thresholds[(vectors, distance.__name__)]
    weightFilter = [[0, t] for t in th]
    threshCl = {
             'algorithm': ['closeness'],
             'weightFilter': weightFilter,
             'weighted': [True, False],
             #'center': ['mean'],
             'center': ['centrality-index'],
    }
    threshComm = {
             'algorithm': ['communicability'],
             'weightFilter': weightFilter,
             'weighted': [False],
             #'center': ['mean'],
             'center': ['centrality-index'],
    }
    # Graph building without thresholding
    nothresh = {
            'algorithm': ['closeness'],
            'weightFilter': None, 'weighted': True,
            #'center': ['mean'],
            'center':['centrality-index'],
    }
    p = jp(fp(basic), fp(threshCl)) + jp(fp(basic), fp(threshComm)) \
        + jp(fp(basic), fp(nothresh))
    p = IdList(jp(p, docSelectParams))
    # label parameter set as either 'world' or 'corpus'
    if vectors in ['word2vec', 'glove']: vecLabel = 'world'
    elif vectors in  ['tf-idf', 'probability']: vecLabel = 'corpus'
    p.id = 'graph_params_%s_vectors' % vecLabel
    return p


def testGraphThresholds(distance, th, label=''):
    basic = {'type':'graph', 'vectors': None,
             'threshold': [50]}
    weightFilter = [ [0, t] for t in th ]
    dist = { 'distance': distance,
             'algorithm': ['closeness', 'communicability'],
             'weightFilter': weightFilter,
             'weighted': [True, False],
             'center': ['median'], }
    p = jp(fp(basic), fp(dist))
    p = IdList(p)
    p.id = 'graph_test_thresholds_%s_%s' % (distance.__name__, label)
    return p

def testEigenCentrality(dist=cosine, method='eigen_centrality'):
    basic = {'type':'graph', 'vectors': 'tf-idf',
             'threshold': [50]}
    if dist == cosine: weightFilter = [0.92056, 0.94344, 0.95661, 0.97246, 0.98525, 0.99478, ]
    elif dist == l2: weightFilter = [1.35688, 1.37364, 1.38319, 1.39460, 1.40374, 1.41052, ]
    elif dist == l1: weightFilter = [9.52592, 10.83310, 12.14663, 14.50215, 17.43981, 20.59608, ]
    weightFilter = [ [0, t] for t in weightFilter ]
    thr = { 'distance': dist,
             'algorithm': [method],
             'weightFilter': weightFilter,
             'weighted': [True, False],
             'center': ['median'], }
    nonthr = { 'distance': dist, 'algorithm': [method],
            'weightFilter': None, 'weighted': True,
            'center': ['median'], }
    p = jp(fp(basic), fp(thr)) + jp(fp(basic), fp(nonthr))
    p = IdList(p)
    p.id = 'graph_test_%s'%method
    return p

def densityParams(vectors):
    if vectors in ['world', 'model', 'models1']:
        # world vectors are max. 300 in size, so they need
        # to be reduced to a dimension << 100
        dimReduce = [None, 5, 10, 20]
    elif vectors == 'corpus':
        dimReduce = [None, 5, 10, 20, 50, 100]
    basic = {
                'type':'density',
                'covariance': ['diag', 'spherical'],
                #'covariance': 'spherical',
                'dimReduce': dimReduce,
                'center': 'mean',
                'threshold': [50]
            }
    basic = IdList(fp(basic))
    basic.id = 'density_params'
    basic = assignVectors(basic, vectors)
    return basic

def matrixParams():
    params = {
              'type': 'matrix',
              'method': ['mu', 'mu0', 'mu1'],
               }
    p = IdList(jp(fp(params), docSelectParams))
    p.id = 'matrix_params'
    return p

def bestInter4Params():
    params = [
        # {'distance': cosine, 'weighted': False, 'center': 'median', 'algorithm': 'clustering',
        #  'vectors': 'glove', 'threshold': 50, 'weightFilter': [0, 0.16095], 'type': 'graph'},
        # {'distance': cosine, 'weighted': True, 'center': 'mean', 'algorithm': 'communicability',
        #  'vectors': 'probability', 'threshold': 50, 'weightFilter': [0, 0.95097], 'type': 'graph'},
        # {'distance': cosine, 'weighted': True, 'center': 'mean', 'algorithm': 'communicability',
        #  'vectors': 'tf-idf', 'threshold': 50, 'weightFilter': [0, 0.95661], 'type': 'graph'},
        # {'distance': cosine, 'weighted': False, 'center': 'mean', 'algorithm': 'communicability',
        #  'threshold': 50, 'weightFilter': [0, 0.95097], 'vectors': 'probability', 'type': 'graph'},
        # {'distance': cosine, 'weighted': True, 'center': 'mean', 'algorithm': 'clustering',
        #  'threshold': 50, 'weightFilter': None, 'vectors': 'probability', 'type': 'graph'}
        {'distance': cosine, 'weighted': False, 'center': 'median', 'algorithm': 'closeness',
         'threshold': 50, 'weightFilter': [0, 0.48094], 'vectors': 'models1', 'type': 'graph'},
        {'distance': l1, 'weighted': False, 'center': 'median', 'algorithm': 'communicability',
         'vectors': 'models1', 'threshold': 50, 'weightFilter': [0,5.03135], 'type': 'graph'}
    ]
    p = IdList(params)
    p.id = 'iter4_best'
    return p

def bestIter3WordCoh():
    params = [  #{'type':'tfidf_coherence'}
                { 'type':'c_v', 'standard': True, 'index':'wiki_standard'},
                {'index': 'wiki_docs', 'type': 'c_v', 'windowSize': 5, 'standard': False},
                {'index': 'wiki_docs', 'type': 'c_v', 'windowSize': 50, 'standard': False},
                { 'type':'c_p', 'standard': True, 'index':'wiki_standard'},
                {'index': 'wiki_docs', 'type': 'c_p', 'windowSize': 20, 'standard': False},
                {'index': 'wiki_docs', 'type': 'c_p', 'windowSize': 0, 'standard': False}
            ]
    params = IdList(params)
    params.id = 'best_wordcoh'
    return params

def assignVectors(params, vectors, tfidf=False):
    '''
    Assigning 'vectors' parameter to each member of params,
     for each possible value from the set of values,
     set of values is either 'corpus' or 'world' vectors.
    '''
    if vectors == 'world': vec = ['word2vec', 'glove']
    elif vectors == 'word2vec': vec = ['word2vec']
    elif vectors == 'glove': vec = ['glove']
    elif vectors == 'corpus': vec = ['tf-idf', 'probability']
    elif vectors == 'tf-idf': vec = ['tf-idf']
    elif vectors == 'probability': vec = ['probability']
    elif vectors == 'model': vec = ['model']
    elif vectors == 'models1': vec = ['models1']
    else: raise Exception('invalid vectors')
    newp = IdList()
    for p in params:
        for vp in vec:
            np = p.copy()
            np['vectors'] = vp
            if (vp == 'glove' or vp == 'word2vec') and tfidf:
                np['tfidf'] = True
            newp.append(np)
    newp.id = params.id + '_%s_vectors' % vectors
    return newp

def matrixParams():
    params = {
              'type': 'matrix',
              #'method': ['mu', 'mu0', 'mu1'],
              'method': ['mu1'],
             }
    p = IdList(jp(fp(params), docSelectParams))
    p.id = 'matrix_params'
    return p

dev, test = devTestSplit2()
dev0, test0 = iter0DevTestSplit()
testFolder = '/datafast/doc_topic_coherence/experiments/test/'

def testParallel(parallel):
    params = assignVectors(testDistanceParams(), 'corpus')
    params.id += ('_parallel[%s]'%parallel)
    print params.id
    tse = TSE(paramSet=params, scorerBuilder=DocCoherenceScorer, parallel=parallel,
              ltopics=dev, posClass=['theme', 'theme_noise'], folder=testFolder, cache=False)
    tse.run()
    tse.printResults()
    #tse.significance(0.05)

def evaluateParams(action, params, vectors=None, dataset=dev, tfidf=False):
    if vectors: params = assignVectors(params, vectors, tfidf=tfidf)
    print params.id
    tse = TSE(paramSet=params, scorerBuilder=DocCoherenceScorer, ltopics=dataset,
              posClass=['theme', 'theme_noise'], folder=testFolder, cache=True)
    if action == 'run': tse.run()
    elif action == 'print': tse.printResults()
    elif action == 'signif': tse.significance(0.05)

if __name__ == '__main__':
    #evaluateParams('print', prodGraphParams('tf-idf', cosine), vectors=None, dataset=dev)
    #evaluateParams('run', prodGraphParams('tf-idf', l2), vectors=None, dataset=dev)
    #evaluateParams('run', prodGraphParams('probability', cosine), vectors=None, dataset=dev)
    #evaluateParams('run', prodGraphParams('probability', l2), vectors=None, dataset=dev)
    #evaluateParams('run', bestInter4Params(), None, test)
    #evaluateParams('run', bestIter3WordCoh(), None, test)
    evaluateParams('run', testDistanceParams(), 'glove', test, tfidf=True)
    #evaluateParams('run', testGraphParams(), 'model', test)
    #evaluateParams('run', densityParams('models1'), None, test)
