from doc_topic_coh.applications.theme_coverage.dataset import *
from doc_topic_coh.applications.theme_coverage.theme_count import \
    AggregateThemeCount, TopicDistEquality, sortByCoherence, themeCount, plotThemeCounts
from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer
from pytopia.measure.topic_distance import cosine

from doc_topic_coh.evaluations.iteration5.best_models import bestParamsDoc, \
    palmettoCp, bestDocCohModel, docCohBaseline

def plotTopicCoverage(modelParams, topics, numRandom=5, rseed=102, simThresh=0.5, maxx=None,
                      cache='/datafast/doc_topic_coherence/experiments/iter5_coherence/function_cache'):
    from copy import copy
    topics = copy(topics)
    if not isinstance(modelParams, list): modelParams = [modelParams]
    thCount = AggregateThemeCount(TopicDistEquality(cosine, simThresh), verbose=False)
    # evaluate counts for models
    cohCounts = []
    for param in modelParams:
        coh = DocCoherenceScorer(cache=cache, **param)()
        topics = sortByCoherence(topics, coh)
        cohCounts.append(thCount(topics))
    # evaluate random counts
    from random import seed, shuffle
    seed(rseed)
    rndCounts = []
    for i in range(numRandom):
        shuffle(topics)
        rndcnt = thCount(topics)
        rndCounts.append(rndcnt)
    figId = 'themeCounts_numRand[%d]_simThresh[%g]' % (numRandom, simThresh)
    plotThemeCounts(cohCounts, rndCounts, maxx=maxx, figid=figId)

topics_nomix = allTopicsNoMix()
topics_mixempty = allTopicsMixedEmpty()

def plotBestDocParams():
    for p in bestParamsDoc():
        print p
        plotTopicCoverage(p, topics_nomix, numRandom=10, maxx=165)
        plotTopicCoverage(p, topics_mixempty, numRandom=10, maxx=205)

def plotAllModels(topics, numRandom=5, simThresh=0.5):
    topGraphDocCohsTest = [
        {'distance': 'cosine', 'weighted': False, 'center': 'mean', 'algorithm': 'communicability',
         'vectors': 'tf-idf', 'threshold': 50, 'weightFilter': [0, 0.92056], 'type': 'graph'},
        {'distance': 'l2', 'weighted': False, 'center': 'mean', 'algorithm': 'communicability', 'vectors': 'tf-idf',
         'threshold': 50, 'weightFilter': [0, 1.37364], 'type': 'graph'}
        # {'distance': 'l2', 'weighted': False, 'center': 'mean', 'algorithm': 'communicability',
        #  'vectors': 'tf-idf', 'threshold': 50, 'weightFilter': [0, 1.35688], 'type': 'graph'},
        # {'distance': 'cosine', 'weighted': False, 'center': 'mean', 'algorithm': 'communicability',
        #  'vectors': 'probability', 'threshold': 50, 'weightFilter': [0, 0.95097], 'type': 'graph'},
        # {'distance': 'cosine', 'weighted': False, 'center': 'mean', 'algorithm': 'communicability',
        #  'vectors': 'tf-idf', 'threshold': 50, 'weightFilter': [0, 0.94344], 'type': 'graph'}
    ]
    params = topGraphDocCohsTest
    params.extend([docCohBaseline, palmettoCp])
    plotTopicCoverage(params, topics, numRandom=numRandom, simThresh=simThresh)

if __name__ == '__main__':
    #plotAllModels(topics_nomix)
    #plotAllModels(topics_mixempty, numRandom=20, simThresh=0.5)
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        plotAllModels(topics_mixempty, numRandom=20, simThresh=thresh)
    #plotTopicCoverage(bestDocCohModel, topics_nomix)
    #plotTopicCoverage(bestDocCohModel, topics_mixempty)
    #plotTopicCoverage(palmettoCp, topics_nomix)
    #plotTopicCoverage(palmettoCp, topics_mixempty)
    #plotTopicCoverage(docCohBaseline, topics_nomix, numRandom=20)
    #plotTopicCoverage(docCohBaseline, topics_mixempty)
    #plotBestDocParams()




