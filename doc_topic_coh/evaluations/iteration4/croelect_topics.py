from doc_topic_coh.resources import pytopia_context

from doc_topic_coh.evaluations.iteration4.best_params import *
from doc_topic_coh.dataset.croelect_dataset import allTopics
from doc_topic_coh.resources.croelect_resources.croelect_resources import corpusId, dictId, text2tokensId
from doc_topic_coh.evaluations.tools import flattenParams as fp, joinParams as jp

croelectTopics = allTopics()
croelectParams = [{ 'corpus':corpusId, 'text2tokens':text2tokensId, 'dict':dictId }]

def bestParamsDocCroelect(type=None, vectors=None, blines=True):
    p = {}
    p[('word', 0)] = [{ 'type':'tfidf_coherence' }]
    p[('distance', 'corpus')] = [
    ]
    p[('distance', 'corpus')] = [
        {'distance': cosine, 'center': 'mean', 'vectors': 'probability', 'exp': 2.0,
         'threshold': 25, 'type': 'avg-dist'}
    ]
    # p[('distance', 'world')] = [
    #     {'distance': cosine, 'center': 'median', 'vectors': 'glove',
    #      'exp': 1.0, 'threshold': 50, 'type': 'variance'}
    # ]
    # p[('distance', 'models1')] = [
    #     {'distance': l2, 'center': 'mean', 'vectors': 'models1', 'exp': 1.0,
    #      'threshold': 50, 'type': 'avg-dist'}
    # ]
    p[('graph', 'corpus')] = [
        {'distance': cosine, 'weighted': True, 'center': 'mean',
         'algorithm': 'communicability', 'vectors': 'probability', 'threshold': 25,
         'weightFilter': [0, 0.95462], 'type': 'graph'}
    ]
    # p[('graph', 'models1')] = [
    #     {'distance': l1, 'weighted': False, 'center': 'median',
    #      'algorithm': 'communicability', 'vectors': 'models1', 'threshold': 50,
    #      'weightFilter': [0, 5.03135], 'type': 'graph'}
    # ]
    # p[('graph', 'world')] = [
    #     {'distance': cosine, 'weighted': False, 'center': 'median',
    #      'algorithm': 'clustering', 'vectors': 'glove', 'threshold': 50,
    #      'weightFilter': [0, 0.16095], 'type': 'graph'}
    # ]
    p[('gauss', 'corpus')] = [
        {'center': 'mean', 'vectors': 'tf-idf', 'covariance': 'spherical',
         'dimReduce': None, 'threshold': 25, 'type': 'density'}
    ]
    # p[('gauss', 'models1')] = [
    #     {'center': 'median', 'vectors': 'models1', 'covariance': 'spherical',
    #      'dimReduce': None, 'threshold': 50, 'type': 'density'}
    # ]
    # p[('gauss', 'world')] = [
    #     {'center': 'median', 'vectors': 'glove', 'dimReduce': 5, 'covariance': 'spherical',
    #      'tfidf': True, 'threshold': 50, 'type': 'density'},
    #     # staro, prije tfidf opcije
    #     # {'center': 'median', 'vectors': 'word2vec', 'covariance': 'diag',
    #     #  'dimReduce': 20, 'threshold': 50, 'type': 'density'}
    # ]
    pl = []
    if type and vectors:
        pl = p[(type, vectors)]
        pl.id += '_%s_%s' % (type, vectors)
    else:
        for par in p.itervalues(): pl.extend(par)
    if blines: pl += docudistBlineParam()
    pl = IdList(jp(pl, croelectParams))
    pl.id = 'best_doc_croelect'
    return pl

gridDistances = [cosine, l1, l2]
docSelectParams = { 'threshold': [10, 25, 50] }
docSelectParams = fp(docSelectParams)
def distanceParams(distance=None, center=None):
    dist = distance if distance else gridDistances
    center = center if center else ['mean', 'median']
    params = {'type': ['variance', 'avg-dist'],
              'vectors': ['probability', 'tf-idf'],
              'distance': dist,
              'center': center,
              'exp': [1.0, 2.0], }
    p = IdList(jp(jp(fp(params), docSelectParams), croelectParams))
    p.id = 'distance_params_croelect'
    return p

def densityParams():
    basic = {
                'type':'density',
                'covariance': ['diag', 'spherical'],
                'center': ['mean', 'median'],
                'dimReduce': [None, 5, 10, 20, 50, 100],
                'vectors': ['probability', 'tf-idf']
            }
    basic = IdList(jp(jp(fp(basic), docSelectParams), croelectParams))
    basic.id = 'density_params_croelect'
    return basic


if __name__ == '__main__':
    #experiment(bestParamsDocCroelect(blines=True), 'run', topics=croelectTopics, cache=False)
    #experiment(distanceParams(), 'run', topics=croelectTopics, cache=False)
    experiment(densityParams(), 'run', topics=croelectTopics, cache=False)
    #for p in bestParamsDocCroelect(blines=True): print p