from doc_topic_coh.resources import pytopia_context

from doc_topic_coh.evaluations.iteration6.best_models import *
from doc_topic_coh.dataset.croelect_dataset import allTopics, labeledTopics
from doc_topic_coh.resources.croelect_resources.croelect_resources import corpusId, dictId, text2tokensId
from doc_topic_coh.evaluations.tools import flattenParams as fp, joinParams as jp

croelectTopics = allTopics()
croelectParams = [{ 'corpus':corpusId, 'text2tokens':text2tokensId, 'dict':dictId }]
model1Topics = labeledTopics(['croelect_model1'])
model2Topics = labeledTopics(['croelect_model2'])
model3Topics = labeledTopics(['croelect_model3'])
model12Topics = labeledTopics(['croelect_model1', 'croelect_model2'])
model123Topics = labeledTopics(['croelect_model1', 'croelect_model2', 'croelect_model3'])
alltopics = labeledTopics(['croelect_model1', 'croelect_model2', 'croelect_model3', 'croelect_model4'])

def bestParamsDocCroelect(blines=True, top=None):
    p = [
        {'distance': cosine, 'weighted': False, 'center': 'mean',
         'algorithm': 'communicability', 'vectors': 'tf-idf',
         'threshold': 50, 'weightFilter': [0, 0.93172], 'type': 'graph'},
        {'distance': cosine, 'center': 'mean', 'vectors': 'probability', 'exp': 1.0,
         'threshold': 50, 'type': 'variance'},
        {'center': 'mean', 'vectors': 'tf-idf', 'covariance': 'spherical',
         'dimReduce': None, 'threshold': 50, 'type': 'density'},
        {'distance': l1, 'weighted': True, 'center': 'mean', 'algorithm': 'clustering',
         'vectors': 'glove-cro-avg', 'threshold': 50, 'weightFilter': [0, 7.54620], 'type': 'graph'},
        {'distance': cosine, 'center': 'mean', 'vectors': 'glove-cro',
         'exp': 1.0, 'threshold': 25, 'type': 'variance'},
        {'center': 'mean', 'vectors': 'word2vec-cro-avg', 'covariance': 'spherical',
         'dimReduce': 10, 'threshold': 25, 'type': 'density'},
    ]
    pl = IdList()
    for par in p: pl.append(par)
    if blines: pl += docudistBlineParam()
    pl = IdList(jp(pl, croelectParams))
    pl.id = 'best_doc_croelect'
    if top:
        for par in pl: par['threshold'] = top
        pl.id += ('_top%d' % top)
    return pl

def palmettoBestCrowiki():
    '''
    Measures from the article "Exploring the Space of Topic Coherence Measures",
    configured to use croatian wikipedia for co-occurrence counts.
    '''
    params = [
                { 'type':'npmi', 'standard': False, 'index': 'crowiki_palmetto_index', 'windowSize': 10},
                { 'type':'uci', 'standard': False, 'index': 'crowiki_palmetto_index', 'windowSize': 10},
                { 'type':'c_a', 'standard': False, 'index': 'crowiki_palmetto_index', 'windowSize': 5},
                { 'type':'c_v', 'standard': False, 'index': 'crowiki_palmetto_index', 'windowSize': 110},
                { 'type':'c_p', 'standard': False, 'index': 'crowiki_palmetto_index', 'windowSize': 70},
            ]
    p = IdList(params);
    p.extend(docudistBlineParam())
    p.id = 'palmetto_best_crowiki_docbline'
    return p

def palmettoBestCroCorpus():
    '''
    Measures from the article "Exploring the Space of Topic Coherence Measures",
    configured to use croatian wikipedia for co-occurrence counts.
    '''
    params = [
                { 'type':'npmi', 'standard': False, 'index': 'croelect_palmetto_index', 'windowSize': 10},
                { 'type':'uci', 'standard': False, 'index': 'croelect_palmetto_index', 'windowSize': 10},
                { 'type':'c_a', 'standard': False, 'index': 'croelect_palmetto_index', 'windowSize': 5},
                { 'type':'c_v', 'standard': False, 'index': 'croelect_palmetto_index', 'windowSize': 110},
                { 'type':'c_p', 'standard': False, 'index': 'croelect_palmetto_index', 'windowSize': 70},
            ]
    p = IdList(params);
    p.append(bestParamsDocCroelect()[0])
    print p
    p.id = 'palmetto_best_crocorpus_docbline'
    return p

def palmettoUspolBestCroelect(bline=False):
    best = {}
    best['uci'] = [
        {'index': 'croelect_palmetto_index', 'type': 'uci', 'windowSize': 5, 'standard': False}
            ]
    best['npmi'] = [
        {'index': 'croelect_palmetto_index', 'type': 'npmi', 'windowSize': 5, 'standard': False}
            ]
    best['c_a'] = [
        {'index': 'croelect_palmetto_index', 'type': 'c_a', 'windowSize': 20, 'standard': False}
            ]
    best['c_v'] = [
        {'index': 'croelect_palmetto_index', 'type': 'c_v', 'windowSize': 0, 'standard': False}
            ]
    best['c_p'] = [
        {'index': 'croelect_palmetto_index', 'type': 'c_p', 'windowSize': 0, 'standard': False}
            ]
    p = IdList()
    for m in best:
        p += best[m]
    p.id = 'palmetto_uspol_best_croelect'
    if bline:
        p.append(bestParamsDocCroelect()[0])
        p.id += '_blined'
    pl = IdList(jp(p, croelectParams))
    pl.id = p.id
    return pl


gridDistances = [cosine, l1, l2]
docSelectParams = { 'threshold': [10, 25, 50] }
docSelectParams = fp(docSelectParams)
# def distanceParams(distance=None, center=None):
#     dist = distance if distance else gridDistances
#     center = center if center else ['mean', 'median']
#     params = {'type': ['variance', 'avg-dist'],
#               'vectors': ['probability', 'tf-idf'],
#               'distance': dist,
#               'center': center,
#               'exp': [1.0, 2.0], }
#     p = IdList(jp(jp(fp(params), docSelectParams), croelectParams))
#     p.id = 'distance_params_croelect'
#     return p
#
# def densityParams():
#     basic = {
#                 'type':'density',
#                 'covariance': ['diag', 'spherical'],
#                 'center': ['mean', 'median'],
#                 'dimReduce': [None, 5, 10, 20, 50, 100],
#                 'vectors': ['probability', 'tf-idf']
#             }
#     basic = IdList(jp(jp(fp(basic), docSelectParams), croelectParams))
#     basic.id = 'density_params_croelect'
#     return basic

croThresholds = {
    ('tf-idf','cosine'): [0.93172, 0.95288, 0.96521, 0.97884, 0.98909, 0.99639, ] ,
    ('tf-idf','l2'): [1.36508, 1.38049, 1.38939, 1.39917, 1.40648, 1.41166, ],
    ('tf-idf','l1'): [11.30759, 12.51893, 13.68560, 15.78645, 18.28902, 21.00582, ],
    ('probability','cosine'): [0.89012, 0.93276, 0.95462, 0.97487, 0.98792, 0.99612, ],
    ('probability','l2'): [0.11922, 0.12929, 0.13854, 0.15582, 0.17922, 0.20827, ],
    ('probability','l1'): [1.86245, 1.89926, 1.92199, 1.95040, 1.97360, 1.99029, ],
    ('word2vec','cosine'): [0.27147, 0.34379, 0.41284, 0.53914, 0.68173, 0.82151, ],
    ('word2vec','l2'): [475.81082, 584.88586, 699.39337, 922.47156, 1252.02527, 1721.08252, ],
    ('word2vec','l1'): [6565.63867, 8061.64111, 9636.98633, 12706.73730, 17234.92188, 23676.13281, ],
    ('word2vec-avg', 'l2'): [3.14411, 3.51542, 3.87957, 4.50022, 5.26021, 6.11886, ],
    ('word2vec-avg', 'l1'): [43.37675, 48.50849, 53.46702, 61.97827, 72.42890, 84.26587, ],
    ('glove', 'cosine'): [0.08100, 0.09782, 0.11514, 0.15029, 0.20064, 0.26623, ],
    ('glove', 'l2'): [62.39438, 77.37844, 93.24452, 126.50224, 181.59282, 270.50101, ],
    ('glove', 'l1'): [860.78955, 1067.40625, 1285.85107, 1742.93213, 2495.15991, 3710.93481, ],
    ('glove-avg', 'l2'): [0.39617, 0.43631, 0.47386, 0.54599, 0.64200, 0.76677, ],
    ('glove-avg', 'l1'): [5.48346, 6.03018, 6.55145, 7.54620, 8.87531, 10.59987, ],
}

def croelectize(p):
    '''
    Modify parameters of a coherence measure to work with croelect dataset,
    modifying resource-related params
    :param p: map of param name -> param value
    :return: copy of p with added and modified param values
    '''
    from copy import copy, deepcopy
    p = deepcopy(p)
    # basic resources
    croelectResParams = {'corpus': corpusId, 'text2tokens': text2tokensId, 'dict': dictId}
    for k,v in croelectResParams.iteritems(): p[k] = v
    if 'weightFilter' in p:
        # this can change for emb, so record now
        vecs, dist = p['vectors'], p['distance'].__name__
    # embedding vectors names
    if 'vectors' in p:
        vp = p['vectors']
        if 'word2vec' in vp or 'glove' in vp:
            if 'avg' in vp: p['vectors'] = vp.replace('avg', 'cro-avg')
            else: p['vectors'] = vp + '-cro'
    # edge thresholds
    if 'weightFilter' in p:
        from doc_topic_coh.evaluations.iteration6.doc_based_coherence import thresholds
        threshold = p['weightFilter'][1]
        th = thresholds[(vecs, dist)]
        croTh = croThresholds[(vecs, dist)]
        found = False
        for i, t in enumerate(th):
            if t == threshold:
                p['weightFilter'][1] = croTh[i]
                found = True
                break
        if not found: raise Exception('threshold not found: %g, %s, %s'%(threshold, vecs, dist))
    return p

def croelectizeParamset(paramset):
    cp = IdList(); cp.id = paramset.id + '_croelectized'
    for p in paramset: cp.append(croelectize(p))
    return cp

def evalTopValues(algo, vectors, plot=False, action='eval'):
    from doc_topic_coh.evaluations.iteration6.doc_based_coherence import \
        graphParams, distanceParams, densityParams
    if algo == 'distance':
        experiment(distanceParams(vectors),
                   action=action, evalTopics=alltopics, evalPerc=0.95, plotEval=plot)
    elif algo == 'graph':
        if vectors == 'corpus':
            experiment(graphParams('tf-idf', cosine),
                       action=action, evalTopics=alltopics, evalPerc=0.95, plotEval=plot)
        elif vectors == 'world':
            experiment(graphParams('word2vec', cosine),
                       action=action, evalTopics=alltopics, evalPerc=0.95, plotEval=plot)
    elif algo == 'gauss':
        experiment(densityParams(vectors),
                   action=action, evalTopics=alltopics, evalPerc=0.95, plotEval=plot)

if __name__ == '__main__':
    #experiment(bestParamsDocCroelect(blines=True), 'run', topics=croelectTopics, cache=False)
    #experiment(distanceParams(), 'run', topics=croelectTopics, cache=False)
    # experiment(palmettoUspolBestCroelect(), action='print', sigThresh=0.0,
    #            topics=croelectTopics, posClass=['theme', 'theme_noise'])
    # experiment(bestParamsDocCroelect(top=None), action='signif', sigThresh=0.0,
    #            topics=alltopics, posClass=['theme', 'theme_noise'], confInt=False,
    #            correct=None)
    #            scoreInd=[3, 0, 1, 2, 4, 5, 6], correct=None)
    #  experiment(palmettoUspolBestCroelect(bline=True), action='signif', sigThresh=0.0,
    #             topics=alltopics, posClass=['theme', 'theme_noise'])
    experiment(palmettoBestCroCorpus(), action='signif', sigThresh=0.0,
               topics=alltopics, posClass=['theme', 'theme_noise'], confInt=False,
               correct=None)
    #for p in bestParamsDocCroelect(blines=True): print p
    #evalTopValues('graph', 'world')