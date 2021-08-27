from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer
from doc_topic_coh.evaluations.tools import flattenParams as fp, joinParams as jp
from doc_topic_coh.dataset.topic_splits import devTestSplit
from doc_topic_coh.evaluations.experiment import IdList, TopicScoringExperiment as TSE
from pytopia.measure.topic_distance import cosine, l1, l2

def palmettoOptWikiParams():
    '''
    Params for optimizing palmetto measures with wiki corpus
    preprocessed with same text2tokens as the original GtAR corpus.
    '''
    all = { 'type':[ 'uci', 'npmi', 'c_a', 'c_p', 'c_v' ],
            'standard': False }
    windows = { 'index':'wiki_docs', 'windowSize': [ 5,10,20,50,100 ] }
    allNonWin = { 'type':['uci', 'npmi', 'c_p', 'c_v'],
            'standard': False }
    boolean = [{'index':'wiki_docs', 'windowSize':0 }]
    p = IdList(jp(fp(all), fp(windows)) + jp(fp(allNonWin), boolean))
    p.id = 'palmetto_grid_wiki'
    return p

def word2vecOptUspolParams():
    '''
    Params for optimizing palmetto measures with wiki corpus
    preprocessed with same text2tokens as the original GtAR corpus.
    '''
    all = { 'type': 'pairwise_word2vec_uspol' }
    cbow = { 'cbow': [0, 1] }
    vecsize = { 'vecsize': [10, 20, 50] }
    window =  { 'window': [3, 5] }
    dist = { 'distance': [l1, l2, cosine] }
    p = IdList(jp(fp(all), jp(fp(dist), jp(fp(cbow), jp(fp(vecsize), fp(window)) )) ))
    p.id = 'word2vec_uspol_grid'
    return p

def palmettoOptUspolParams():
    '''
    Params for optimizing palmetto measures with uspol corpus.
    :return:
    '''
    p = IdList()
    for typ in [ 'uci', 'npmi', 'c_a', 'c_p', 'c_v' ]:
        p.extend(palmettoOptUspolParamsPerType(typ))
    p.id = 'palmetto_grid_uspol'
    return p

def palmettoOptUspolParamsPerType(type):
    all = { 'type': type, 'standard': False }
    wsizes = [5, 10, 20, 50, 100]
    if type != 'c_a': wsizes.append(0)
    windows = { 'index':'uspol_palmetto_index', 'windowSize': wsizes }
    p = IdList(jp([all], fp(windows)))
    p.id = 'palmetto_uspol_params_%s' % type
    return p

dev, test = devTestSplit()
expFolder = '/datafast/doc_topic_coherence/experiments/iter5_coherence/'

from doc_topic_coh.evaluations.iteration5.doc_based_coherence import experiment

if __name__ == '__main__':
    #print word2vecOptUspolParams()
    experiment(word2vecOptUspolParams(), action='print')
    #experiment(word2vecOptUspolParams(), action='print')
    #experiment(palmettoOptWikiParams(), action='print')
    #experiment(palmettoOptUspolParams(), action='print')
