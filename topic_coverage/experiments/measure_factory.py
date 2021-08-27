from pytopia.measure.topic_distance import cosine, l1norm, hellinger
from pytopia.topic_functions.cached_function import CachedFunction
from topic_coverage.experiments.correlation.measure_correlation_utils import Topic2ModelDist, Topic2ModelMatch
from topic_coverage.topicmatch.ctc_matcher import CtcModelCoverage
from topic_coverage.topicmatch.supervised_matching import TopicmatchModelCoverage, \
    TopicmatchModelPrecision, optSupervisedMatcher, optSupervisedMatcherPheno

from topic_coverage.settings import function_cache_folder
from os.path import join

supmodelcovCache = join(function_cache_folder, 'sup_model_coverage')
ctcmodelcovCache = join(function_cache_folder, 'ctc_model_coverage')
topicmatchCache = join(function_cache_folder, 'topic_match')

def supervisedModelCoverage(corpus='uspol', strict=True, typ=None, covCache=True, matchCache=True):
    tmc = TopicmatchModelCoverage(supervisedTopicMatcher(corpus, strict, cached=matchCache, typ=typ))
    if covCache == True:
        return CachedFunction(tmc, supmodelcovCache, saveEvery=1, verbose=True)
    elif isinstance(covCache, str) and covCache != '':
        return CachedFunction(tmc, covCache, saveEvery=1, verbose=True)
    else: return tmc

def supervisedModelPrecision(corpus='uspol', strict=True, typ=None, covCache=True, matchCache=True):
    tmprec = TopicmatchModelPrecision(supervisedTopicMatcher(corpus, strict, cached=matchCache, typ=typ))
    if covCache == True:
        return CachedFunction(tmprec, supmodelcovCache, saveEvery=1, verbose=True)
    elif isinstance(covCache, str) and covCache != '':
        return CachedFunction(tmprec, covCache, saveEvery=1, verbose=True)
    else: return tmprec

def supervisedTopicMatcher(corpus='uspol', strict=True, cached=False, typ=None):
    from topic_coverage.topicmatch.supervised_best_models import optModelV2
    from topic_coverage.topicmatch.supervised_matching import SupervisedTopicMatcher
    if typ == 'nocos': feats = 'core1nocos'
    else: feats = 'core1'
    m = SupervisedTopicMatcher(optModelV2(corpus, strict, feats))
    if not cached: return m
    else:
        if cached == True:
            return CachedFunction(m, topicmatchCache, saveEvery=20000, verbose=True)
        else:
            return CachedFunction(m, cached, saveEvery=20000, verbose=True)

def ctcModelCoverage(strict=False, topicDist=cosine, cached=True):
    if topicDist == cosine:
        if strict: mn, mx, intervals = 0.0, 0.4, 50
        else: mn, mx, intervals = 0.0, 1.0, 50
    elif topicDist == l1norm: mn, mx, intervals = 0.0, 2.0, 50
    elif topicDist == hellinger: mn, mx, intervals = 0.0, 1.0, 50
    else: mn, mx, intervals = None, None, None
    ctc = CtcModelCoverage(topicDist, mn, mx, intervals)
    if cached:
        if cached == True:
            return CachedFunction(ctc, ctcmodelcovCache, saveEvery=1, verbose=True)
        else:
            return CachedFunction(ctc, cached, saveEvery=1, verbose=True)
    else: return ctc

def cachedTopic2ModelDist(refmodel, topicDist=cosine):
    t2md = Topic2ModelDist(refmodel, topicDist)
    return CachedFunction(t2md, ctcmodelcovCache, saveEvery=50, verbose=True)

def cachedTopic2ModelMatch(refmodel, matcher):
    t2mm = Topic2ModelMatch(refmodel, matcher)
    return CachedFunction(t2mm, supmodelcovCache, saveEvery=50, verbose=True)

if __name__ == '__main__':
    pass