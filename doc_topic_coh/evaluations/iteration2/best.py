from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer, scorersFromParams
from doc_topic_coh.evaluations.tools import topicMeasureAuc, \
    flattenParams as fp, joinParams as jp
from doc_topic_coh.dataset.topic_splits import iter0DevTestSplit, devTestSplit2
from pytopia.measure.topic_distance import cosine, l1, l2
from doc_topic_coh.evaluations.experiment import IdDict, IdList, \
    TopicScoringExperiment as TSE

def paramsBestAndBaselines(select=['palmetto', 'bline_other',
                                   'corpus_vectors', 'word2vec_vectors']):
    ############# baselines
    if 'palmetto' in select or 'bline_other' in select:
        #todo update with additional paramters for new palmetto coherence class
        blinePalmetto = []
        if 'palmetto' in select:
            blinePalmetto = palmettoWikiBest()+palmettoUspolBest()
        blineOther = []
        if 'bline_other' in select:
            blineOther = { 'type': ['text_distribution', 'pairwise_word2vec', 'tfidf_coherence'] }
        bline = blinePalmetto+fp(blineOther)
    else: bline = []
    ############# best algorithms
    if 'corpus_vectors' in select:
        ## corpus vectors
        bestDistC = [
            {'distance': cosine, 'center': 'median', 'vectors': 'probability', 'exp': 1.0, 'threshold': 50, 'type': 'avg-dist'},
            {'distance': cosine, 'center': 'median', 'vectors': 'tf-idf', 'exp': 1.0, 'threshold': 100, 'type': 'avg-dist'},
            {'distance': cosine, 'center': 'median', 'vectors': 'probability', 'exp': 1.0, 'threshold': 100, 'type': 'variance'}
                     ]
        bestGraphC = [
            {'distance': cosine, 'weighted': True, 'center': 'mean', 'algorithm': 'communicability', 'vectors': 'probability', 'threshold': 50, 'weightFilter': [0, 0.95], 'type': 'graph'},
            {'distance': cosine, 'weighted': False, 'center': 'mean', 'algorithm': 'closeness', 'vectors': 'probability', 'threshold': 50, 'weightFilter': [0, 0.95], 'type': 'graph'}
        ]
        bestDensityC = [
            {'scoreMeasure': 'll', 'vectors': 'tf-idf', 'covariance': 'spherical', 'dimReduce': None, 'threshold': 50,
             'type': 'density'},
            {'scoreMeasure': 'll', 'vectors': 'probability', 'covariance': 'spherical', 'dimReduce': 50,
             'threshold': 50, 'type': 'density'}
        ]
        bestMatrixC = [
            {'threshold': 50, 'vectors': 'probability', 'type': 'matrix', 'method': 'mu0'},
            {'threshold': 100, 'vectors': 'tf-idf', 'type': 'matrix', 'method': 'mu'},
                     ]
        bestCorpus = bestDistC+bestMatrixC+bestGraphC+bestDensityC
    else: bestCorpus = []

    ## word2vec vectors
    if 'word2vec_vectors':
        bestMatrixW2V = [
            {'threshold': 50, 'vectors': 'word2vec', 'type': 'matrix', 'method': 'mu'},
            {'threshold': 100, 'vectors': 'word2vec', 'type': 'matrix', 'method': 'mu1'}
        ]
        bestDistW2V = [
            {'distance': cosine, 'center': 'mean', 'vectors': 'word2vec', 'exp': 1.0, 'threshold': 50, 'type': 'avg-dist'},
            {'distance': cosine, 'center': 'median', 'vectors': 'word2vec', 'exp': 1.0, 'threshold': 50, 'type': 'variance'}
        ]
        bestGraphW2V = [
            {'distance': cosine, 'weighted': True, 'center': 'mean', 'algorithm': 'closeness', 'vectors': 'word2vec', 'threshold': 50, 'weightFilter': [0,0.9], 'type': 'graph'},
            {'distance': cosine, 'weighted': True, 'center': 'mean', 'algorithm': 'closeness', 'vectors': 'word2vec', 'threshold': 50, 'weightFilter': None, 'type': 'graph'}
        ]
        bestDensWord2V = [
            {'scoreMeasure': 'll', 'vectors': 'word2vec', 'covariance': 'diag', 'dimReduce': 20, 'threshold': 50,
             'type': 'density'},
            {'scoreMeasure': 'll', 'vectors': 'word2vec', 'covariance': 'diag', 'dimReduce': 50, 'threshold': 100,
             'type': 'density'}
        ]
        bestWord2Vec = bestDistW2V+bestMatrixW2V+bestGraphW2V+bestDensWord2V
    else: bestWord2Vec = []
    bestAlgo = bestCorpus + bestWord2Vec
    p = IdList(bestAlgo + bline)
    p.id = 'best_algos_and_baselines'
    return p

def palmettoWikiBest(type=None, index=None):
    best = {}
    # best['umass'] = [
    #         { 'type':'umass', 'standard': True, 'index':'wiki_standard'},
    #         {'index': 'wiki_paragraphs', 'type': 'umass', 'windowSize': 0, 'standard': False},
    #         #{'index': 'wiki_docs', 'type': 'umass', 'windowSize': 20, 'standard': False}
    #         ]
    best['npmi'] = [
                #{'type':'npmi', 'standard': True, 'index':'wiki_standard'},
                {'index': 'wiki_docs', 'type': 'npmi', 'windowSize': 20, 'standard': False},
                #{'index': 'wiki_docs', 'type': 'npmi', 'windowSize': 100, 'standard': False}
            ]
    best['uci'] = [
                #{ 'type':'uci', 'standard': True, 'index':'wiki_standard' },
                {'index': 'wiki_docs', 'type': 'uci', 'windowSize': 0, 'standard': False},
                #{'index': 'wiki_docs', 'type': 'uci', 'windowSize': 100, 'standard': False}
            ]
    best['c_a'] = [
                #{ 'type':'c_a', 'standard': True, 'index':'wiki_standard'},
                {'index': 'wiki_docs', 'type': 'c_a', 'windowSize': 5, 'standard': False},
                #{'index': 'wiki_docs', 'type': 'c_a', 'windowSize': 10, 'standard': False},
            ]
    best['c_v'] = [
                #{ 'type':'c_v', 'standard': True, 'index':'wiki_standard'},
                {'index': 'wiki_docs', 'type': 'c_v', 'windowSize': 5, 'standard': False},
                #{'index': 'wiki_docs', 'type': 'c_v', 'windowSize': 50, 'standard': False}
            ]
    best['c_p'] = [
                #{ 'type':'c_p', 'standard': True, 'index':'wiki_standard'},
                {'index': 'wiki_docs', 'type': 'c_p', 'windowSize': 20, 'standard': False},
                #{'index': 'wiki_docs', 'type': 'c_p', 'windowSize': 0, 'standard': False}
            ]
    if type:
        p = IdList(best[type])
        p.id = 'palmetto_best_%s'%type
    else:
        p = IdList()
        for m in best:
            p += best[m]
        p.id = 'palmetto_best_all'
    return p

def palmettoUspolBest(type=None):
    best = {}
    best['uci'] = [
        {'index': 'uspol_palmetto_index', 'type': 'uci', 'windowSize': 0, 'standard': False},
        #{'index': 'uspol_palmetto_index', 'type': 'uci', 'windowSize': 10, 'standard': False}
            ]
    best['npmi'] = [
        {'index': 'uspol_palmetto_index', 'type': 'npmi', 'windowSize': 0, 'standard': False},
        #
        #{'index': 'uspol_palmetto_index', 'type': 'npmi', 'windowSize': 10, 'standard': False}
            ]
    # best['umass'] = [
    #     {'index': 'uspol_palmetto_index', 'type': 'umass', 'windowSize': 10, 'standard': False},
    #     #{'index': 'uspol_palmetto_index', 'type': 'umass', 'windowSize': 0, 'standard': False}
    #         ]
    best['c_a'] = [
        {'index': 'uspol_palmetto_index', 'type': 'c_a', 'windowSize': 20, 'standard': False},
        #{'index': 'uspol_palmetto_index', 'type': 'c_a', 'windowSize': 100, 'standard': False},
            ]
    best['c_v'] = [
        {'index': 'uspol_palmetto_index', 'type': 'c_v', 'windowSize': 0, 'standard': False},

            ]
    best['c_p'] = [
        {'index': 'uspol_palmetto_index', 'type': 'c_p', 'windowSize': 0, 'standard': False},
        #{'index': 'uspol_palmetto_index', 'type': 'c_p', 'windowSize': 50, 'standard': False}
            ]
    if type:
        p = IdList(best[type])
        p.id = 'palmetto_uspol_best_%s'%type
    else:
        p = IdList()
        for m in best:
            p += best[m]
        p.id = 'palmetto_uspol_best_all'
    return p

dev, test = devTestSplit2()
expFolder = '/datafast/doc_topic_coherence/experiments/iter2_coherence/'
cacheFolder = '/datafast/doc_topic_coherence/experiments/iter2_coherence/function_cache'

def experimentAllTest():
    #selection = ['corpus_vectors', 'word2vec_vectors']
    #selection = ['bline_other']
    tse = TSE(paramSet=paramsBestAndBaselines(),
              scorerBuilder=DocCoherenceScorer,  ltopics=test, posClass=['theme', 'theme_noise'],
              folder=expFolder, cache=True)
    #tse.run()
    #tse.printResults(confInt=True)
    tse.significance(scoreInd=[0,2,6,11,20], correct=True, threshold=None)

def experimentPalmettoBest(type=None):
    tse = TSE(paramSet=palmettoWikiBest(type),
              scorerBuilder=DocCoherenceScorer,  ltopics=test, posClass=['theme', 'theme_noise'],
              folder=expFolder, cache=True)
    tse.run()

def experimentPalmettoUspolBest():
    tse = TSE(paramSet=palmettoUspolBest(),
              scorerBuilder=DocCoherenceScorer,  ltopics=test, posClass=['theme', 'theme_noise'],
              folder=expFolder, cache=True)
    tse.run()

if __name__ == '__main__':
    experimentAllTest()
    #experimentPalmettoBest('c_a')
    #experimentPalmettoBest('c_p')
    #experimentPalmettoBest('c_v')
    #experimentPalmettoBest('umass')
    #experimentPalmettoBest('uci')
    #experimentPalmettoBest('npmi')
    #experimentPalmettoUspolBest()
