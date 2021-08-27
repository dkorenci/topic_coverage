from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer, scorersFromParams
from doc_topic_coh.evaluations.tools import topicMeasureAuc, \
    flattenParams as fp, joinParams as jp
from doc_topic_coh.dataset.topic_splits import devTestSplit2, topicSplit
from pytopia.measure.topic_distance import cosine, l1, l2
from doc_topic_coh.evaluations.experiment import IdDict, IdList, \
    TopicScoringExperiment as TSE

from doc_topic_coh.evaluations.iteration5.doc_based_coherence import devTestSplit
from doc_topic_coh.evaluations.iteration5.best_models import \
    docudistBlineParam, bestDocCohModel
from doc_topic_coh.evaluations.iteration5.word_based_coherence import word2vecOptUspolParams

dev, test = devTestSplit()
expFolder = '/datafast/doc_topic_coherence/experiments/iter5_coherence_testing/'

def bestParamsDoc(type=None, vectors=None, blines=True):
    p = {}
    p[('distance', 'corpus')] = [
    ]
    p[('distance', 'corpus')] = [
        {'distance': cosine, 'center': 'mean', 'vectors': 'probability', 'exp': 2.0,
         'threshold': 50, 'type': 'avg-dist'}
    ]
    p[('distance', 'world')] = [
        {'distance': cosine, 'center': 'median', 'vectors': 'glove',
         'exp': 1.0, 'threshold': 50, 'type': 'variance'}
    ]
    # p[('distance', 'models1')] = [
    #     {'distance': l2, 'center': 'mean', 'vectors': 'models1', 'exp': 1.0,
    #      'threshold': 50, 'type': 'avg-dist'}
    # ]
    p[('graph', 'corpus')] = [
        {'distance': cosine, 'weighted': True, 'center': 'mean',
         'algorithm': 'communicability', 'vectors': 'probability', 'threshold': 50,
         'weightFilter': [0, 0.95097], 'type': 'graph'}
    ]
    # p[('graph', 'models1')] = [
    #     {'distance': l1, 'weighted': False, 'center': 'median',
    #      'algorithm': 'communicability', 'vectors': 'models1', 'threshold': 50,
    #      'weightFilter': [0, 5.03135], 'type': 'graph'}
    # ]
    p[('graph', 'world')] = [
        {'distance': cosine, 'weighted': False, 'center': 'median',
         'algorithm': 'clustering', 'vectors': 'glove', 'threshold': 50,
         'weightFilter': [0, 0.16095], 'type': 'graph'}
    ]
    p[('gauss', 'corpus')] = [
        {'center': 'mean', 'vectors': 'tf-idf', 'covariance': 'spherical',
         'dimReduce': None, 'threshold': 50, 'type': 'density'}
    ]
    # p[('gauss', 'models1')] = [
    #     {'center': 'median', 'vectors': 'models1', 'covariance': 'spherical',
    #      'dimReduce': None, 'threshold': 50, 'type': 'density'}
    # ]
    p[('gauss', 'world')] = [
        {'center': 'median', 'vectors': 'glove', 'dimReduce': 5, 'covariance': 'spherical',
         'tfidf': True, 'threshold': 50, 'type': 'density'},
        # staro, prije tfidf opcije
        # {'center': 'median', 'vectors': 'word2vec', 'covariance': 'diag',
        #  'dimReduce': 20, 'threshold': 50, 'type': 'density'}
    ]
    pl = IdList(); pl.id = 'best_doc'
    if type and vectors:
        pl = p[(type, vectors)]
        pl.id += '_%s_%s' % (type, vectors)
    else:
        for par in p.itervalues(): pl.extend(par)
    if blines: pl += docudistBlineParam()
    return pl


def experiment(params, topics, action='run', vectors=None, confInt=False):
    print params.id
    print 'num params', len(params)
    tse = TSE(paramSet=params, scorerBuilder=DocCoherenceScorer,
              ltopics=topics, posClass=['theme', 'theme_noise'],
              folder=expFolder, cache=True)
    if action == 'run': tse.run()
    elif action == 'print': tse.printResults(confInt=confInt)
    elif action == 'signif': tse.significance(threshold=-1)
    else: print 'specified action not defined'

def testBalance(split, action='run'):
    dev, test = split
    print '************* DEV *************'
    print dev.id
    experiment(bestParamsDoc(), dev, action=action)
    print '************* TEST *************'
    print test.id
    experiment(bestParamsDoc(), test, action=action)

def testBoxplot():
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1,1)
    x = [1,2,3,4,5]
    axes.boxplot(x)
    axes.scatter([1]*len(x), x, alpha=0.5)
    axes.plot(1, 3, 'ro')
    plt.show()

def bestAndBline():
    p = IdList()
    p.append(bestDocCohModel)
    p.extend(docudistBlineParam())
    p.id = 'bestdoc_and_blinedoc'
    return p

if __name__ == '__main__':
    #testBalance(topicSplit(), 'run')
    #testBoxplot()
    experiment(bestAndBline(), test, 'signif')
    experiment(word2vecOptUspolParams(), dev, 'print')
    pass

