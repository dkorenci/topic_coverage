from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer
from doc_topic_coh.evaluations.tools import flattenParams as fp, joinParams as jp
from doc_topic_coh.dataset.topic_splits import devTestSplit
from pytopia.measure.topic_distance import cosine, l1, l2
from doc_topic_coh.evaluations.experiment import IdDict, IdList, \
    TopicScoringExperiment as TSE

def palmettoBaselineParams():
    # standard - measure with original palmetto wiki index
    # nonstandard - measure with custom uspol-preprocessed wiki index
    params = [
                #{ 'type':'npmi', 'standard': True, 'index':'wiki_standard'},
                { 'type':'npmi', 'standard': False, 'index': 'wiki_docs', 'windowSize': 10},
                #{ 'type':'uci', 'standard': True, 'index':'wiki_standard'},
                { 'type':'uci', 'standard': False, 'index': 'wiki_docs', 'windowSize': 10},
                #{ 'type':'c_a', 'standard': True, 'index':'wiki_standard'},
                { 'type':'c_a', 'standard': False, 'index': 'wiki_docs', 'windowSize': 5},
                #{ 'type':'c_v', 'standard': True, 'index':'wiki_standard'},
                { 'type':'c_v', 'standard': False, 'index': 'wiki_docs', 'windowSize': 110},
                #{ 'type':'c_p', 'standard': True, 'index':'wiki_standard'},
                { 'type':'c_p', 'standard': False, 'index': 'wiki_docs', 'windowSize': 70},
            ]
    p = IdList(params); p.id = 'palmetto_baseline'
    return p

# word coherence selected as representative, for various experiments
palmettoCp = { 'type':'c_p', 'standard': False, 'index': 'wiki_docs', 'windowSize': 70}

def palmettoWikiBest(type=None):
    best = {}
    best['npmi'] = [
                {'index': 'wiki_docs', 'type': 'npmi', 'windowSize': 100, 'standard': False}
            ]
    best['uci'] = [
                {'index': 'wiki_docs', 'type': 'uci', 'windowSize': 0, 'standard': False}
            ]
    best['c_a'] = [
                {'index': 'wiki_docs', 'type': 'c_a', 'windowSize': 5, 'standard': False}
            ]
    best['c_v'] = [
                {'index': 'wiki_docs', 'type': 'c_v', 'windowSize': 100, 'standard': False}
            ]
    best['c_p'] = [
                {'index': 'wiki_docs', 'type': 'c_p', 'windowSize': 100, 'standard': False}
            ]
    if type:
        p = IdList(best[type])
        p.id = 'palmetto_wiki_best_%s'%type
    else:
        p = IdList()
        for m in best:
            p += best[m]
        p.id = 'palmetto_wiki_best'
    return p

def palmettoUspolBest(type=None):
    best = {}
    best['uci'] = [
        {'index': 'uspol_palmetto_index', 'type': 'uci', 'windowSize': 5, 'standard': False}
            ]
    best['npmi'] = [
        {'index': 'uspol_palmetto_index', 'type': 'npmi', 'windowSize': 5, 'standard': False}
            ]
    best['c_a'] = [
        {'index': 'uspol_palmetto_index', 'type': 'c_a', 'windowSize': 20, 'standard': False}
            ]
    best['c_v'] = [
        {'index': 'uspol_palmetto_index', 'type': 'c_v', 'windowSize': 0, 'standard': False}
            ]
    best['c_p'] = [
        {'index': 'uspol_palmetto_index', 'type': 'c_p', 'windowSize': 0, 'standard': False}
            ]
    if type:
        p = IdList(best[type])
        p.id = 'palmetto_uspol_best_%s'%type
    else:
        p = IdList()
        for m in best:
            p += best[m]
        p.id = 'palmetto_uspol_best'
    return p

def word2vecUspolBest():
    params = [
        {'cbow': 0, 'window': 5, 'type': 'pairwise_word2vec_uspol', 'vecsize': 10},
        {'cbow': 1, 'window': 5, 'type': 'pairwise_word2vec_uspol', 'vecsize': 10},
    ]
    p = IdList(params)
    p.id = 'word2vec_uspol_best'
    return p

def wordCohOptimized():
    p = IdList([{'type': 'tfidf_coherence'}])
    p += palmettoWikiBest()
    p += palmettoUspolBest()
    p.id = 'wordcoh_optimized'
    return p

def docudistBlineParam():
    p = IdList([ {'type':'text_distribution'} ])
    p.id = 'docu_dist_baseline'
    return p

docCohBaseline = {'type':'text_distribution'}

def wordcohBaselines(docBaseline=False):
    '''
    Word coherences from original papers, not optimized for doc-coherence.
    '''
    params = [ # removed word2vec-coh
        #{'type': 'pairwise_word2vec_wiki', 'cbow':1},
        #{'type': 'pairwise_word2vec_wiki', 'cbow':0},
    ]
    p = IdList(params)
    p.extend(palmettoBaselineParams())
    p.id = 'wordcoh_baselines'
    if docCohBaseline:
        p.append(docCohBaseline)
        p.id += '_doc_blined'
    return p

def wordcohWord2vecBaselines():
    '''
    Word coherences from original papers, not optimized for doc-coherence.
    '''
    params = [ # only the word2vec-coh with higher score is included
        {'type': 'pairwise_word2vec_wiki', 'cbow':1, 'distance': l1 },
        {'type': 'pairwise_word2vec_wiki', 'cbow':0, 'distance': l1 },
    ]
    p = IdList(params)
    p.extend(palmettoBaselineParams())
    p.id = 'wordcoh_word2vec_baselines'
    return p


def wordcohBaselinesDocBased():
    '''
    wordcohBaselines() coherences, plus doc-coherence baseline as the
    first method, for significance testing.
    '''
    p = wordcohBaselines()
    p += docudistBlineParam()
    p.id += '_docbaselined'
    print p.id
    return p

bestDocCohModel = { 'distance': cosine, 'weighted': False, 'center': 'mean',
         'algorithm': 'communicability', 'vectors': 'tf-idf',
         'threshold': 50, 'weightFilter': [0, 0.92056], 'type': 'graph' }

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
        {'distance': cosine, 'weighted': False, 'center': 'mean',
         'algorithm': 'communicability', 'vectors': 'glove',
         'threshold': 25, 'weightFilter': [0, 0.0909], 'type': 'graph'}
    ]
    p[('gauss', 'corpus')] = [
        {'center': 'mean', 'vectors': 'tf-idf', 'covariance': 'spherical',
         'dimReduce': None, 'threshold': 50, 'type': 'density'}
    ]
    p[('gauss', 'world')] = [
        {'center': 'mean', 'vectors': 'glove', 'covariance': 'diag', 'dimReduce': 5,
         'threshold': 25, 'type': 'density'}
    ]
    pl = IdList(); pl.id = 'best_doc'
    if type and vectors:
        pl = p[(type, vectors)]
        pl.id += '_%s_%s' % (type, vectors)
    else:
        for par in p.itervalues(): pl.extend(par)
    if blines: pl += docudistBlineParam()
    return pl

from doc_topic_coh.evaluations.iteration5.doc_based_coherence import experiment
dev, test = devTestSplit()

if __name__ == '__main__':
    experiment(bestParamsDoc(), topics=test, action='signif', sigThresh=-0.1)
    #experiment(word2vecUspolBest(), topics=test, action='print')
    #experiment(wordcohBaselinesDocBased(), topics=test, action='run')
    #experiment(wordcohBaselinesDocBased(), topics=test, action='signif', sigThresh=-0.1)
    #experiment(wordcohWord2vecBaselines(), topics=test, action='run')
    #experiment(wordcohBaselines(), topics=test, action='signif', sigThresh=-0.1)
    #experiment(palmettoWikiBest(), topics=test, action='print')
    #experiment(palmettoUspolBest(), topics=test, action='signif', sigThresh=-0.1)
    #experiment(wordCohOptimized(), topics=test, action='signif', sigThresh=-0.1)