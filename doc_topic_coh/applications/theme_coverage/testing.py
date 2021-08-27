from doc_topic_coh.applications.theme_coverage.dataset import *
from doc_topic_coh.applications.theme_coverage.theme_count import \
    AggregateThemeCount, TopicDistEquality, sortByCoherence, themeCount, plotThemeCounts
from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer
from pytopia.measure.topic_distance import cosine


def test1():
    cache = '/datafast/doc_topic_coherence/experiments/iter4_coherence/function_cache'
    cohParams = {'distance': cosine, 'weighted': True, 'center': 'mean',
         'algorithm': 'communicability', 'vectors': 'probability', 'threshold': 50,
         'weightFilter': [0, 0.95097], 'type': 'graph'}
    cohParams = {'distance': cosine, 'center': 'mean', 'vectors': 'probability', 'exp': 2.0,
         'threshold': 50, 'type': 'avg-dist'}
    cohParams = {'index': 'wiki_docs', 'type': 'npmi', 'windowSize': 20, 'standard': False}
    coh = DocCoherenceScorer(cache=cache, **cohParams)()
    tt = allTopicsNoMix()
    tt = sortByCoherence(tt, coh)
    print themeCount(tt)
    from random import seed, shuffle
    seed(102)
    for i in range(5):
        shuffle(tt)
        print themeCount(tt)

def test2():
    cache = '/datafast/doc_topic_coherence/experiments/iter4_coherence/function_cache'
    graphPar = {'distance': cosine, 'weighted': True, 'center': 'mean',
         'algorithm': 'communicability', 'vectors': 'probability', 'threshold': 50,
         'weightFilter': [0, 0.95097], 'type': 'graph'}
    distPar = {'distance': cosine, 'center': 'mean', 'vectors': 'probability', 'exp': 2.0,
         'threshold': 50, 'type': 'avg-dist'}
    wordPar = {'index': 'wiki_docs', 'type': 'npmi', 'windowSize': 20, 'standard': False}
    cohParams = graphPar
    coh = DocCoherenceScorer(cache=cache, **cohParams)()
    #tt = allTopicsNoMix()
    tt = allTopicsMixedEmpty()
    tt = sortByCoherence(tt, coh)
    thCount = AggregateThemeCount(TopicDistEquality(cosine, 0.5), verbose=False)
    #print '*****************SORTED**********************'
    cohCounts = thCount(tt)
    from random import seed, shuffle
    seed(102)
    rndCounts = []
    for i in range(5):
        #print '*****************RANDOM**********************'
        shuffle(tt)
        rndcnt = thCount(tt)
        rndCounts.append(rndcnt)
    plotThemeCounts([cohCounts], rndCounts)

if __name__ == '__main__':
    test2()




