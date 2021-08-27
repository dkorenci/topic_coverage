from pytopia.measure.topic_distance import cosine, l1, l2
from doc_topic_coh.evaluations.experiment import IdList

from doc_topic_coh.evaluations.iteration2.best import palmettoUspolBest, palmettoWikiBest
from doc_topic_coh.evaluations.iteration4.doc_based_coherence import assignVectors

def bestParamsWords():
    otherWordCoh = [
        { 'type':'pairwise_word2vec' },
        { 'type':'tfidf_coherence' }
    ]
    p = IdList(palmettoWikiBest() + palmettoUspolBest() + otherWordCoh)
    p.id = 'best_words'
    return p

def topWordCohParam():
    return [ {'index': 'uspol_palmetto_index', 'type': 'c_p', 'windowSize': 0, 'standard': False} ]

def docudistBlineParam(): return [ {'type':'text_distribution'} ]

def palmettoOriginalParams():
    params = [
                { 'type':'npmi', 'standard': True, 'index':'wiki_standard'},
                { 'type':'uci', 'standard': True, 'index':'wiki_standard'},
                { 'type':'c_a', 'standard': True, 'index':'wiki_standard'},
                { 'type':'c_v', 'standard': True, 'index':'wiki_standard'},
                { 'type':'c_p', 'standard': True, 'index':'wiki_standard'},
            ]
    p = IdList(params); p.id = 'palmetto_original'
    return p

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
    p[('distance', 'models1')] = [
        {'distance': l2, 'center': 'mean', 'vectors': 'models1', 'exp': 1.0,
         'threshold': 50, 'type': 'avg-dist'}
    ]
    p[('graph', 'corpus')] = [
        {'distance': cosine, 'weighted': True, 'center': 'mean',
         'algorithm': 'communicability', 'vectors': 'probability', 'threshold': 50,
         'weightFilter': [0, 0.95097], 'type': 'graph'}
    ]
    p[('graph', 'models1')] = [
        {'distance': l1, 'weighted': False, 'center': 'median',
         'algorithm': 'communicability', 'vectors': 'models1', 'threshold': 50,
         'weightFilter': [0, 5.03135], 'type': 'graph'}
    ]
    p[('graph', 'world')] = [
        {'distance': cosine, 'weighted': False, 'center': 'median',
         'algorithm': 'clustering', 'vectors': 'glove', 'threshold': 50,
         'weightFilter': [0, 0.16095], 'type': 'graph'}
    ]
    p[('gauss', 'corpus')] = [
        {'center': 'mean', 'vectors': 'tf-idf', 'covariance': 'spherical',
         'dimReduce': None, 'threshold': 50, 'type': 'density'}
    ]
    p[('gauss', 'models1')] = [
        {'center': 'median', 'vectors': 'models1', 'covariance': 'spherical',
         'dimReduce': None, 'threshold': 50, 'type': 'density'}
    ]
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

from doc_topic_coh.dataset.topic_splits import devTestSplit2, iter0DevTestSplit, allTopics
from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer
from doc_topic_coh.evaluations.experiment import TopicScoringExperiment as TSE

dev0, test0 = iter0DevTestSplit()
dev, test = devTestSplit2()
expFolder = '/datafast/doc_topic_coherence/experiments/iter4_coherence/'
alltopics = allTopics()

def experiment(params, action='run', vectors=None, topics=test, confInt=False, threshold=None,
               cache=True):
    if vectors: params = assignVectors(params, vectors)
    print params.id
    print 'num params', len(params)
    tse = TSE(paramSet=params, scorerBuilder=DocCoherenceScorer,
              ltopics=topics, posClass=['theme', 'theme_noise'],
              folder=expFolder, cache=cache)
    if action == 'run': tse.run()
    elif action == 'print': tse.printResults(confInt=confInt)
    elif action == 'signif': tse.significance(threshold=threshold)
    else: print 'specified action not defined'

def tableDocumentCoherences1BestDoc(action='run', topics=test):
    experiment(bestParamsDoc(), action, topics=topics, confInt=True)

def tableWordCoherences1BestWord():
    experiment(bestParamsWords(), 'print', dataset='dev', confInt=True)

if __name__ == '__main__':
    #tableDocumentCoherences1BestDoc('print', alltopics)
    #tableWordCoherences1BestWord()
    #experiment(palmettoOriginalParams(), 'print', topics=test, confInt=True)
    experiment(bestParamsDoc(), 'signif', topics=test)