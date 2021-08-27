from doc_topic_coh.evaluations.iteration5.best_models import *

bestDocCohModel = { 'distance': cosine, 'weighted': False, 'center': 'mean',
         'algorithm': 'communicability', 'vectors': 'tf-idf',
         'threshold': 50, 'weightFilter': [0, 0.92056], 'type': 'graph' }

def bestGraphWorld():
    params = [ # only the word2vec-coh with higher score is included
        {'distance': l1, 'weighted': True, 'center': 'mean', 'algorithm': 'clustering', 'vectors': 'glove-avg',
         'threshold': 25, 'weightFilter': [0, 14.31248], 'type': 'graph'},
        {'distance': l2, 'weighted': True, 'center': 'mean', 'algorithm': 'clustering', 'vectors': 'glove-avg',
         'threshold': 25, 'weightFilter': [0, 1.05602], 'type': 'graph'},
        {'distance': l2, 'weighted': True, 'center': 'mean', 'algorithm': 'clustering', 'vectors': 'word2vec-avg',
         'threshold': 25, 'weightFilter': [0, 0.46448], 'type': 'graph'},
        {'distance': cosine, 'weighted': False, 'center': 'mean', 'algorithm': 'communicability', 'vectors': 'glove',
         'threshold': 25, 'weightFilter': [0, 0.0909], 'type': 'graph'},
        {'distance': l2, 'weighted': False, 'center': 'mean', 'algorithm': 'num_connected', 'vectors': 'glove-avg',
         'threshold': 25, 'weightFilter': [0, 1.05602], 'type': 'graph'},
        {'distance': l1, 'weighted': True, 'center': 'mean', 'algorithm': 'clustering', 'vectors': 'word2vec-avg',
         'threshold': 25, 'weightFilter': [0, 6.42045], 'type': 'graph'},
        {'distance': cosine, 'weighted': True, 'center': 'mean', 'algorithm': 'clustering', 'vectors': 'glove',
         'threshold': 25, 'weightFilter': [0, 0.05979], 'type': 'graph'},
        {'distance': cosine, 'weighted': False, 'center': 'mean', 'algorithm': 'closeness', 'vectors': 'glove',
         'threshold': 25, 'weightFilter': [0, 0.0909], 'type': 'graph'},
        {'distance': l2, 'weighted': True, 'center': 'mean', 'algorithm': 'clustering', 'vectors': 'word2vec-avg',
         'threshold': 25, 'weightFilter': [0, 0.42444], 'type': 'graph'},
        {'distance': cosine, 'weighted': False, 'center': 'mean', 'algorithm': 'clustering', 'vectors': 'glove',
         'threshold': 25, 'weightFilter': [0, 0.12111], 'type': 'graph'},
    ]
    p = IdList(params)
    p.id = 'best_doc_graph_world'
    return p

def bestParamsDoc(type=None, vectors=None, blines=True):
    p = {}
    p[('distance', 'corpus')] = [
        {'distance': cosine, 'center': 'mean', 'vectors': 'probability', 'exp': 1.0,
         'threshold': 50, 'type': 'variance'}
    ]
    p[('distance', 'world')] = [
        {'distance': cosine, 'center': 'mean', 'vectors': 'glove',
         'exp': 1.0, 'threshold': 25, 'type': 'variance'}
    ]
    p[('graph', 'corpus')] = [
        {'distance': cosine, 'weighted': False, 'center': 'mean',
         'algorithm': 'communicability', 'vectors': 'tf-idf',
         'threshold': 50, 'weightFilter': [0, 0.92056], 'type': 'graph'}
    ]
    p[('graph', 'world')] = [
        {'distance': l1, 'weighted': True, 'center': 'mean', 'algorithm': 'clustering',
         'vectors': 'glove-avg', 'threshold': 25, 'weightFilter': [0, 14.31248], 'type': 'graph'}
    ]
    p[('gauss', 'corpus')] = [
        {'center': 'mean', 'vectors': 'tf-idf', 'covariance': 'spherical',
         'dimReduce': None, 'threshold': 50, 'type': 'density'}
    ]
    p[('gauss', 'world')] = [
        {'center': 'mean', 'vectors': 'word2vec-avg', 'covariance': 'spherical',
         'dimReduce': 10, 'threshold': 25, 'type': 'density'}
    ]
    pl = IdList(); pl.id = 'best_doc'
    if type and vectors:
        pl = p[(type, vectors)]
        pl.id += '_%s_%s' % (type, vectors)
    else:
        for par in p.itervalues(): pl.extend(par)
    if blines: pl += docudistBlineParam()
    return pl

from doc_topic_coh.evaluations.iteration6.doc_based_coherence import experiment
dev, test = devTestSplit()

if __name__ == '__main__':
    # experiment(bestParamsDoc(), topics=test, action='signif', sigThresh=-0.1,
    #            correct=None)
               #scoreInd=[4, 0,1,2,3,5,6], correct=None)
    #experiment(bestGraphWorld(), topics=test, action='run', sigThresh=-0.1)
    #experiment(bestParamsDoc(), topics=test, action='signif', sigThresh=-0.1)
    #experiment(word2vecUspolBest(), topics=test, action='print')
    #experiment(wordcohBaselinesDocBased(), topics=test, action='run')
    #experiment(wordcohBaselinesDocBased(), topics=test, action='signif', sigThresh=-0.1)
    #experiment(wordcohWord2vecBaselines(), topics=test, action='run')
    experiment(wordcohBaselines(True), topics=test, action='signif', sigThresh=-0.1,
               correct=None)
    #experiment(palmettoWikiBest(), topics=test, action='print')
    #experiment(palmettoUspolBest(), topics=test, action='signif', sigThresh=-0.1)
    #experiment(wordCohOptimized(), topics=test, action='signif', sigThresh=-0.1)