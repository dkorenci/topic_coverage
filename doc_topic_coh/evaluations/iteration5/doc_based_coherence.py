from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer
from doc_topic_coh.evaluations.tools import flattenParams as fp, joinParams as jp
from doc_topic_coh.dataset.topic_splits import devTestSplit
from pytopia.measure.topic_distance import cosine, l1, l2
from doc_topic_coh.evaluations.experiment import IdDict, IdList, \
    TopicScoringExperiment as TSE

docSelectParams = { 'threshold': [10, 25, 50, 100] }
docSelectParams = fp(docSelectParams)

gridDistances = [cosine, l1, l2]

def distanceParams(distance=None):
    dist = distance if distance else gridDistances
    params = {
              'type': ['variance', 'avg-dist'],
              'vectors': None,
              'distance': dist,
              'center': 'mean',
              'exp': 1.0
              }
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
}

def thres2perc(vectors, distance, thVal):
    '''
    Converts value of distance threshold to corresponding percentile.
    '''
    # from distance_distribution.py , percentiles for which thresholds were printed
    threshPercs = [0.02, 0.05, 0.1, 0.25, 0.5, 0.75]
    for i, th in enumerate(thresholds[(vectors, distance)]):
        if thVal == th: return threshPercs[i]

def graphParams(vectors, distance):
    basic = { 'type':'graph', 'vectors': vectors, 'distance': distance }
    # Graph building with thresholding
    th = thresholds[(vectors, distance.__name__)]
    weightFilter = [[0, t] for t in th]
    thresh = {
             'algorithm': ['clustering', 'closeness'],
             'weightFilter': weightFilter,
             'weighted': [True, False],
             'center': 'mean',
    }
    threshNonWeighted = {
             'algorithm': ['communicability', 'num_connected'],
             'weightFilter': weightFilter,
             'weighted': False,
             'center': 'mean'
    }
    # Graph building without thresholding
    nothresh = {
            'algorithm': ['closeness', 'mst', 'clustering'],
            'weightFilter': None, 'weighted': True,
            'center': 'mean'
    }
    p = jp(fp(basic), fp(thresh)) + jp(fp(basic), fp(threshNonWeighted)) + jp(fp(basic), fp(nothresh))
    p = IdList(jp(p, docSelectParams))
    # label parameter set as either 'world' or 'corpus'
    if vectors in ['word2vec', 'glove']: vecLabel = 'world'
    elif vectors in  ['tf-idf', 'probability']: vecLabel = 'corpus'
    p.id = 'graph_params_%s_vectors' % vecLabel
    return p

def densityParams(vectors):
    if vectors == 'world':
        # world vectors are max. 300 in size, so they need
        # to be reduced to a dimension << 100
        dimReduce = [None, 5, 10, 20]
    elif vectors == 'corpus':
        dimReduce = [None, 5, 10, 20, 50, 100]
    basic = {
                'type': 'density',
                'covariance': ['diag', 'spherical'],
                'center': 'mean',
                'dimReduce': dimReduce
            }
    basic = IdList(jp(fp(basic), docSelectParams))
    basic.id = 'density_params'
    basic = assignVectors(basic, vectors)
    return basic

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
    if modifyId: newp.id = params.id + '_%s_vectors' % vectors
    else: newp.id = params.id
    return newp

dev, test = devTestSplit()
expFolder = '/datafast/doc_topic_coherence/experiments/iter5_coherence/'

def experiment(params, topics=dev, action='run', vectors=None, confInt=False,
               sigThresh=0.05, evalTopics=None, plotEval=False, evalPerc=None,
               evalThresh=None, th2per=None):
    if vectors: params = assignVectors(params, vectors)
    print params.id
    print 'num params', len(params)
    tse = TSE(paramSet=params, scorerBuilder=DocCoherenceScorer,
              ltopics=topics, posClass=['theme', 'theme_noise'],
              folder=expFolder, cache=True)
    if action == 'run': tse.run()
    elif action == 'print': tse.printResults(confInt=confInt)
    elif action == 'signif': tse.significance(threshold=sigThresh)
    elif action == 'eval':
        tse.evalOnTopics(evalTopics, plot=plotEval, percentile=evalPerc)
    elif action == 'printTop':
        tse.evalOnTopicsPrintTop(evalTopics, thresh=evalThresh, percentile=evalPerc,
                                 th2per=th2per)
    else: print 'specified action not defined'

def runGridDistance(vectors, action='run'):
    experiment(distanceParams(), dev, action, vectors=vectors)

def runGridGraphCorpus(action='run'):
    if action != 'run':
        experiment(graphParams('tf-idf', cosine), action=action)
    else:
        experiment(graphParams('tf-idf', cosine))
        experiment(graphParams('tf-idf', l1))
        experiment(graphParams('tf-idf', l2))
        experiment(graphParams('probability', cosine))
        experiment(graphParams('probability', l1))
        experiment(graphParams('probability', l2))

def runGridGraphWorld(action='run'):
    if action == 'print':
        experiment(graphParams('word2vec', cosine), action='print')
    else:
        experiment(graphParams('word2vec', cosine))
        experiment(graphParams('word2vec', l1))
        experiment(graphParams('word2vec', l2))
        experiment(graphParams('glove', cosine))
        experiment(graphParams('glove', l1))
        experiment(graphParams('glove', l2))

def runGridDensity(vectors, action='run'):
    experiment(densityParams(vectors), action=action)

def evalTopValues(algo, vectors, plot=False):
    if algo == 'distance':
        experiment(distanceParams(), vectors=vectors, action='eval',
                   evalTopics=test, evalPerc=0.9, plotEval=plot)
    elif algo == 'graph':
        if vectors == 'corpus':
            experiment(graphParams('tf-idf', cosine), action='eval',
                       evalTopics=test, evalPerc=0.95, plotEval=plot)
        elif vectors == 'world':
            experiment(graphParams('word2vec', cosine), action='eval',
                       evalTopics=test, evalPerc=0.95, plotEval=plot)
    elif algo == 'gauss':
        experiment(densityParams(vectors), action='eval', evalTopics=test,
                   evalPerc=0.9, plotEval=plot)

def printTopSelected(algo, vectors):
    if algo == 'distance':
        if vectors == 'corpus':
            experiment(distanceParams(), vectors=vectors, action='printTop',
                   evalTopics=test, evalPerc=0.9, evalThresh='median')
        elif vectors == 'world':
            experiment(distanceParams(), vectors=vectors, action='printTop',
                   evalTopics=test, evalPerc=0.9, evalThresh=0.7316)
    elif algo == 'graph':
        if vectors == 'corpus':
            experiment(graphParams('tf-idf', cosine), action='printTop',
                       evalTopics=test, evalPerc=0.95, evalThresh='median', th2per=thres2perc)
        elif vectors == 'world':
            experiment(graphParams('word2vec', cosine), action='printTop',
                       evalTopics=test, evalPerc=0.95, evalThresh='median', th2per=thres2perc)
    elif algo == 'gauss':
        if vectors == 'corpus':
            experiment(densityParams(vectors), action='printTop',
                   evalTopics=test, evalPerc=0.9, evalThresh='median')
        elif vectors == 'world':
            experiment(densityParams(vectors), action='printTop',
                   evalTopics=test, evalPerc=0.9, evalThresh='median')

def evalAllOnTest(plot=False):
    for algo in ['graph', 'distance', 'gauss']:
        for vec in ['corpus', 'world']:
            evalTopValues(algo, vec, plot=plot)

def evalPrintAllOnTest():
    for algo in ['graph', 'distance', 'gauss']:
        for vec in ['corpus', 'world']:
            printTopSelected(algo, vec)


if __name__ == '__main__':
    #evalAllOnTest(plot=False)
    #evalPrintAllOnTest()
    #printTopSelected('graph', 'corpus')
    #evalTopValues('gauss', 'world')
    #runGridGraphCorpus('print')
    runGridGraphWorld('print')
    #runGridDistance('corpus', 'print')
    #runGridDistance('world', 'print')
    #runGridGraphCorpus('print')
    #runGridGraphWorld('print')
    #runGridDensity('corpus', 'print')
    #runGridDensity('world', 'print')
    pass
