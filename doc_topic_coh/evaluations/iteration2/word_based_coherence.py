from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer, scorersFromParams
from doc_topic_coh.evaluations.tools import topicMeasureAuc, \
    flattenParams as fp, joinParams as jp
from doc_topic_coh.dataset.topic_splits import devTestSplit2
from pytopia.measure.topic_distance import cosine, l1, l2
from doc_topic_coh.evaluations.experiment import IdDict, IdList, \
    TopicScoringExperiment as TSE

def palmettoParams():
    all = { 'type':['umass', 'uci', 'npmi', 'c_a', 'c_p', 'c_v'],
            'standard': False }
    windows = { 'index':'wiki_docs', 'windowSize': [5,10,20,50,100] }
    allNonWin = { 'type':['umass', 'uci', 'npmi', 'c_p', 'c_v'],
            'standard': False }
    boolean = {'index': ['wiki_docs','wiki_paragraphs'], 'windowSize':0 }
    p = IdList(jp(fp(all), fp(windows)) + jp(fp(allNonWin), fp(boolean)))
    p.id = 'palmetto_params'
    return p

def palmettoUspolParams(type):
    all = { 'type': type, 'standard': False }
    wsizes = [5, 10, 20, 50, 100]
    if type != 'c_a': wsizes.append(0)
    windows = { 'index':'uspol_palmetto_index', 'windowSize': wsizes }
    p = IdList(jp([all], fp(windows)))
    p.id = 'palmetto_uspol_params_%s' % type
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

dev, test = devTestSplit2()
expFolder = '/datafast/doc_topic_coherence/experiments/iter2_coherence/'

def experimentPalmetto():
    tse = TSE(paramSet=palmettoParams(), scorerBuilder=DocCoherenceScorer,
              ltopics=dev, posClass=['theme', 'theme_noise'], folder=expFolder, cache=True)
    #tse.run()
    tse.printResults()

def experimentPalmettoUspol(type):
    tse = TSE(paramSet=palmettoUspolParams(type), scorerBuilder=DocCoherenceScorer,
              ltopics=dev, posClass=['theme', 'theme_noise'], folder=expFolder, cache=True)
    tse.run()
    #tse.printResults()
    #tse.significance()

def testPalmetto(type, index=None):
    tse = TSE(paramSet=palmettoBest(type, index), scorerBuilder=DocCoherenceScorer,
              ltopics=test, posClass=['theme', 'theme_noise'], folder=expFolder, cache=True)
    #tse.run()
    tse.printResults()
    tse.testSignificance(0, 1, N=100000)

if __name__ == '__main__':
    #experimentPalmetto()
    #experimentPalmettoUspol('uci')
    #experimentPalmettoUspol('npmi')
    #experimentPalmettoUspol('umass')
    experimentPalmettoUspol('c_a')
    #experimentPalmettoUspol('c_p')
    #experimentPalmettoUspol('c_v')
