from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer, scorersFromParams
from doc_topic_coh.evaluations.tools import topicMeasureAuc, \
    flattenParams as fp, joinParams as jp
from doc_topic_coh.dataset.topic_splits import devTestSplit2
from pytopia.measure.topic_distance import cosine, l1, l2, l2squared, lInf, canberra
from doc_topic_coh.evaluations.experiment import IdDict, IdList, \
    TopicScoringExperiment as TSE

docSelectParams = { 'threshold': [10, 25, 50, 100] }
docSelectParams = fp(docSelectParams)

allDistances = [cosine, l1, l2, l2squared, lInf, canberra]
gridDistances = [cosine, l1, l2]

def distanceParams(distance=None, center=None):
    dist = distance if distance else gridDistances
    center = center if center else ['mean', 'median']
    params = {'type': ['variance', 'avg-dist'],
              'vectors': None,
              'distance': dist,
              'center': center,
              'exp': [1.0, 2.0], }
    p = IdList(jp(fp(params), docSelectParams))
    p.id = 'distance_params'
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
    ('model','cosine'): [0.37167, 0.47968, 0.56031, 0.67801, 0.78156, 0.85596, ] ,
    ('model','l2'): [0.39597, 0.45666, 0.51563, 0.62909, 0.78244, 0.96179, ] ,
    ('model','l1'): [3.36978, 3.93406, 4.38349, 5.02163, 5.63806, 6.18251, ],
    ('models1', 'cosine'): [0.37340, 0.48094, 0.56179, 0.68014, 0.78361, 0.85819, ],
    ('models1', 'l2'): [0.39657, 0.45820, 0.51765, 0.63147, 0.78749, 0.97252, ],
    ('models1', 'l1'): [3.37666, 3.93703, 4.38667, 5.03135, 5.65444, 6.20220, ],
    ('word2vec-tfidf', 'cosine'): [0.10869, 0.13288, 0.15741, 0.20256, 0.26224, 0.33065, ],
    ('word2vec-tfidf', 'l2'): [4.91631, 5.35253, 5.78459, 6.62321, 7.78016, 9.19462, ],
    ('word2vec-tfidf', 'l1'): [67.97868, 73.92184, 79.98488, 91.63013, 107.72115, 127.53425, ],
    ('glove-tfidf', 'cosine'): [0.07726, 0.09622, 0.11639, 0.15570, 0.20981, 0.27654, ],
    ('glove-tfidf', 'l2'): [12.37310, 13.59828, 14.82791, 17.21134, 20.49674, 24.43962, ],
    ('glove-tfidf', 'l1'): [165.16138, 180.13828, 194.58994, 221.81903, 257.11478, 295.99744, ],
}
def graphParams(vectors, distance, tfidf=False):
    basic = {'type':'graph', 'vectors': vectors, 'distance': distance}
    if tfidf: basic['tfidf'] = True
    # Graph building with thresholding
    vecLabel = vectors if not tfidf else vectors+'-tfidf'
    th = thresholds[(vecLabel, distance.__name__)]
    weightFilter = [[0, t] for t in th]
    thresh = {
             'algorithm': ['clustering', 'closeness'],
             'weightFilter': weightFilter,
             'weighted': [True, False],
             'center': ['mean', 'median'],
    }
    threshNonWeighted = {
             'algorithm': ['communicability', 'num_connected'],
             'weightFilter': weightFilter,
             'weighted': [False],
             'center': ['mean', 'median'],
    }
    # Graph building without thresholding
    nothresh = {
            'algorithm': ['closeness', 'mst', 'clustering'],
            'weightFilter': None, 'weighted': True,
            'center': ['mean', 'median'],
    }
    p = jp(fp(basic), fp(thresh)) + jp(fp(basic), fp(threshNonWeighted)) + jp(fp(basic), fp(nothresh))
    p = IdList(jp(p, docSelectParams))
    # label parameter set as either 'world' or 'corpus'
    if vectors in ['word2vec', 'glove']: vecLabel = 'world'
    elif vectors in  ['tf-idf', 'probability']: vecLabel = 'corpus'
    else: vecLabel = vectors
    p.id = 'graph_params_%s_vectors' % vecLabel
    return p

def densityParams(vectors):
    if vectors == 'world':
        # world vectors are max. 300 in size, so they need
        # to be reduced to a dimension << 100
        dimReduce = [None, 5, 10, 20]
    elif vectors in ['model', 'models1']:
        dimReduce = [None, 5, 10, 20]
    elif vectors == 'corpus':
        dimReduce = [None, 5, 10, 20, 50, 100]
    basic = {
                'type':'density',
                'covariance': ['diag', 'spherical'],
                'center': ['mean', 'median'],
                'dimReduce': dimReduce
            }
    basic = IdList(jp(fp(basic), docSelectParams))
    basic.id = 'density_params'
    basic = assignVectors(basic, vectors)
    return basic

def matrixParams():
    params = {
              'type': 'matrix',
              #'method': ['mu', 'mu0', 'mu1'],
              'method': ['mu', 'mu0'],
             }
    p = IdList(jp(fp(params), docSelectParams))
    p.id = 'matrix_params'
    return p

def assignVectors(params, vectors, modifyId=True):
    '''
    Assigning 'vectors' parameter to each member of params,
     for each possible value from the set of values,
     set of values is either 'corpus' or 'world' vectors.
    '''
    if vectors == 'world': vec = ['word2vec', 'glove']
    elif vectors == 'corpus': vec = ['tf-idf', 'probability']
    elif vectors == 'model': vec = ['model']
    elif vectors == 'models1': vec = ['models1']
    else: raise Exception('invalid vectors')
    newp = IdList()
    for p in params:
        for vp in vec:
            np = p.copy()
            np['vectors'] = vp
            newp.append(np)
            if vectors == 'world':
                np = np.copy()
                np['tfidf'] = True
                newp.append(np)
    if modifyId: newp.id = params.id + '_%s_vectors' % vectors
    else: newp.id = params.id
    return newp

def countParams():
    dp = len(distanceParams(cosine, 'mean'))*4
    gp = 0
    for vec in ['tf-idf', 'probability', 'word2vec', 'glove']:
        for dist in gridDistances:
            gp += len(graphParams(vec, dist))
    pp = len(densityParams('world'))*4
    pp += len(densityParams('corpus'))*4
    mp = len(matrixParams())*4
    print 'dist %d, graph %d, dens %d, matrix %d' % (dp, gp, pp, mp)
    print 'all', dp+gp+pp+mp

dev, test = devTestSplit2()
expFolder = '/datafast/doc_topic_coherence/experiments/iter4_coherence/'

def experiment(params, action='run', vectors=None, dataset='dev', confInt=False):
    if vectors: params = assignVectors(params, vectors)
    if dataset == 'dev': topics = dev
    else: topics = test
    print params.id
    print 'num params', len(params)
    tse = TSE(paramSet=params, scorerBuilder=DocCoherenceScorer,
              ltopics=topics, posClass=['theme', 'theme_noise'],
              folder=expFolder, cache=True)
    if action == 'run': tse.run()
    elif action == 'print': tse.printResults(confInt=confInt)
    elif action == 'signif': tse.significance(threshold=0.05)
    else: print 'specified action not defined'

def runGridDistanceCorpus(action='run'):
    experiment(distanceParams(), action, vectors='corpus')
    #experiment(distanceParams(cosine, 'mean'), vectors='corpus')
    #experiment(distanceParams(l2, 'mean'), vectors='corpus')
    #experiment(distanceParams(l1, 'mean'), vectors='corpus')
    # experiment(distanceParams(cosine, 'median'), vectors='corpus')
    # experiment(distanceParams(l2, 'median'), vectors='corpus')
    # experiment(distanceParams(l1, 'median'), vectors='corpus')

def runGridDistance(vectors, action='run'):
    experiment(distanceParams(), action, vectors=vectors)

def runGridDistanceWorld():
    experiment(distanceParams(), 'run', vectors='world')
    # experiment(distanceParams(cosine, 'mean'), vectors='world')
    # experiment(distanceParams(l2, 'mean'), vectors='world')
    # experiment(distanceParams(l1, 'mean'), vectors='world')
    # experiment(distanceParams(cosine, 'median'), vectors='world')
    # experiment(distanceParams(l2, 'median'), vectors='world')
    # experiment(distanceParams(l1, 'median'), vectors='world')

def runGridGraphCorpus(action='run'):
    if action == 'print':
        experiment(graphParams('tf-idf', cosine), 'print')
    else:
        experiment(graphParams('tf-idf', cosine))
        experiment(graphParams('tf-idf', l1))
        experiment(graphParams('tf-idf', l2))
        experiment(graphParams('probability', cosine))
        experiment(graphParams('probability', l1))
        experiment(graphParams('probability', l2))

def runGridGraph(vectors, action='run', tfidf=False):
    if action == 'run':
        experiment(graphParams(vectors, cosine, tfidf=tfidf))
        experiment(graphParams(vectors, l2, tfidf=tfidf))
        experiment(graphParams(vectors, l1, tfidf=tfidf))
    else:
        experiment(graphParams(vectors, cosine), action)

def runGridGraphWorld():
    experiment(graphParams('word2vec', cosine), 'print')
    return
    experiment(graphParams('word2vec', cosine))
    experiment(graphParams('word2vec', l1))
    experiment(graphParams('word2vec', l2))
    experiment(graphParams('glove', cosine))
    experiment(graphParams('glove', l1))
    experiment(graphParams('glove', l2))

def runGridDensity(vectors, action='run'):
    experiment(densityParams(vectors), action)

def runGridMatrix(vectors, action='run'):
    experiment(assignVectors(matrixParams(), vectors=vectors), action)

if __name__ == '__main__':
    #runGridDistance('models1', 'signif')
    runGridGraph('probability', 'signif')
    #runGridGraph('models1', 'print')
    #runGridGraph('word2vec', 'run', tfidf=True)
    #runGridGraph('word2vec', 'print')
    #runGridGraph('word2vec', 'print')
    #runGridDensity('corpus', 'print')
    #runGridDensity('world', 'print')
    #runGridDensity('models1', 'print')