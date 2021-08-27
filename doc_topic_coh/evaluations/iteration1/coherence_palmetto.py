from doc_topic_coh.dataset.topic_splits import iter0DevTestSplit
from doc_topic_coh.evaluations.experiment import IdList, \
    TopicScoringExperiment as TSE
from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer
from doc_topic_coh.evaluations.tools import flattenParams as fp, joinParams as jp

def palmettoParams1():
    all = { 'type':['umass', 'uci', 'npmi', 'c_a', 'c_p', 'c_v'],
            'standard': False }
    windows = { 'index':'wiki_docs', 'windowSize': [5,10,20] }
    allNonWin = { 'type':['umass', 'uci', 'npmi', 'c_p', 'c_v'],
            'standard': False }
    boolean = {'index': ['wiki_docs','wiki_paragraphs'], 'windowSize':0 }

    p = IdList(jp(fp(all), fp(windows)) + jp(fp(allNonWin), fp(boolean)))
    p.id = 'palmetto_params'
    return p

def palmettoParams2():
    all = { 'type':['umass', 'uci', 'npmi', 'c_a', 'c_p', 'c_v'],
            'standard': False }
    windows = { 'index':'wiki_docs', 'windowSize': [50,100] }
    p = IdList(jp(fp(all), fp(windows))) #+ jp(fp(allNonWin), fp(boolean)))
    p.id = 'palmetto_params2'
    return p

def palmettoBest(type, index=None):
    best = {}
    best['umass'] = [
                #{ 'type':'umass', 'standard': True, 'index':'wiki_standard'},
                { 'type': 'umass', 'standard': False, 'index': 'wiki_docs', 'windowSize': 0},
                {'index': 'wiki_docs', 'type': 'umass', 'windowSize': 10, 'standard': False},
                {'index': 'wiki_docs', 'type': 'umass', 'windowSize': 5, 'standard': False},
            ]
    best['npmi'] = [
                #{ 'type':'npmi', 'standard': True, 'index':'wiki_standard'},
                { 'type': 'npmi', 'standard': False, 'index': 'wiki_docs', 'windowSize': 10},
                {'index': 'wiki_docs', 'type': 'npmi', 'windowSize': 5, 'standard': False},
                {'index': 'wiki_docs', 'type': 'npmi', 'windowSize': 0, 'standard': False},
            ]
    best['uci'] = [
                #{ 'type':'uci', 'standard': True, 'index':'wiki_standard'},
                {'type': 'uci', 'standard': False, 'index': 'wiki_docs', 'windowSize': 10},
                {'index': 'wiki_docs', 'type': 'uci', 'windowSize': 0, 'standard': False},
                {'index': 'wiki_docs', 'type': 'uci', 'windowSize': 10, 'standard': False},
            ]
    best['c_a'] = [
                #{ 'type':'c_a', 'standard': True, 'index':'wiki_standard'},
                {'type': 'c_a', 'standard': False, 'index': 'wiki_docs', 'windowSize': 5},
                {'index': 'wiki_docs', 'type': 'c_a', 'windowSize': 10, 'standard': False},
                {'index': 'wiki_docs', 'type': 'c_a', 'windowSize': 20, 'standard': False},
            ]
    best['c_v'] = [
                #{ 'type':'c_v', 'standard': True, 'index':'wiki_standard'},
                { 'type': 'c_v', 'standard': False, 'index': 'wiki_docs', 'windowSize': 110},
                {'index': 'wiki_docs', 'type': 'c_v', 'windowSize': 50, 'standard': False},
                {'index': 'wiki_paragraphs', 'type': 'c_v', 'windowSize': 0, 'standard': False}
            ]
    best['c_p'] = [
                #{ 'type':'c_p', 'standard': True, 'index':'wiki_standard'},
                { 'type': 'c_p', 'standard': False, 'index': 'wiki_docs', 'windowSize': 70},
                {'index': 'wiki_docs', 'type': 'c_p', 'windowSize': 5, 'standard': False},
                {'index': 'wiki_docs', 'type': 'c_p', 'windowSize': 100, 'standard': False},
            ]

    p = IdList(best[type])
    p.id = 'palmetto_best_%s_%s'%(type, index)
    if index:
        for param in p: param['index'] = index
    return p

dev, test = iter0DevTestSplit()
expFolder = '/datafast/doc_topic_coherence/experiments/iter1_coherence/'


def experimentPalmetto():
    tse = TSE(paramSet=palmettoParams2(), scorerBuilder=DocCoherenceScorer,
              ltopics=dev, posClass=['theme', 'theme_noise'], folder=expFolder, cache=True)
    #tse.run()
    tse.printResults()

def testPalmetto(type, index=None):
    tse = TSE(paramSet=palmettoBest(type, index), scorerBuilder=DocCoherenceScorer,
              ltopics=test, posClass=['theme', 'theme_noise'], folder=expFolder, cache=True)
    #tse.run()
    tse.printResults()
    tse.testSignificance(0, 1, N=100000)

if __name__ == '__main__': pass
    #experimentPalmetto()
    #testPalmetto('uci', 'uspol_palmetto_index')
    #testPalmetto('npmi', 'uspol_palmetto_index')
    #testPalmetto('umass', 'uspol_palmetto_index')
    #testPalmetto('c_v' , 'uspol_palmetto_index')
    #testPalmetto('c_a', 'uspol_palmetto_index')
    #testPalmetto('c_p', 'uspol_palmetto_index')
