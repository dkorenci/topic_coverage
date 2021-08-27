from pytopia.measure.topic_distance import cosine as cosineDist, klDivZero as KL
from pytopia.topic_functions.cached_function import CachedFunction
from topic_coverage.experiments.correlation.measure_correlation_utils import Topic2ModelDist, Topic2ModelMatch
from topic_coverage.settings import function_cache_folder
from topic_coverage.topicmatch.ctc_matcher import CtcModelCoverage
from topic_coverage.topicmatch.supervised_matching import TopicmatchModelCoverage, \
    optSupervisedMatcher, optSupervisedMatcherPheno

functionCacheFolder = '/datafast/topic_coverage/function_cache/'

def supervisedMatcherV1(corpus='uspol', model='logreg'):
    tmc = TopicmatchModelCoverage(supervisedTopicMatcher(corpus, model=model))
    return CachedFunction(tmc, function_cache_folder, saveEvery=1)

def supervisedModelMatcherV2(corpus='uspol', strict=True):
    tmc = TopicmatchModelCoverage(supervisedTopicMatcherV2(corpus, strict))
    return CachedFunction(tmc, function_cache_folder, saveEvery=1)

def supervisedTopicMatcher(corpus='uspol', cached=False, model='logreg'):
    if corpus == 'uspol': m = optSupervisedMatcher(iter=1, model=model)
    elif corpus == 'pheno': m = optSupervisedMatcherPheno()
    if not cached: return m
    else: return CachedFunction(m, function_cache_folder, saveEvery=10)

def supervisedTopicMatcherV2(corpus='uspol', strict=True, cached=False):
    from topic_coverage.topicmatch.supervised_best_models import optModelV1
    from topic_coverage.topicmatch.supervised_matching import SupervisedTopicMatcher
    m = SupervisedTopicMatcher(optModelV1(corpus, strict))
    if not cached: return m
    else: return CachedFunction(m, function_cache_folder, saveEvery=10)

def ctcMatcherCos(strict=False):
    if strict: mn, mx, intervals = 0, 0.4, 50
    else: mn, mx, intervals = 0, 1, 50
    ctc = CtcModelCoverage(cosineDist, mn, mx, intervals)
    return CachedFunction(ctc, function_cache_folder, saveEvery=1)

def ctcMatcherKL():
    ctc = CtcModelCoverage(KL, 0, 16, 30)
    return CachedFunction(ctc, function_cache_folder, saveEvery=1)

def ctcMatcher(typ='cos', strict=False):
    if typ == 'cos': return ctcMatcherCos(strict)
    elif typ == 'kl': return ctcMatcherKL()

def cachedTopic2ModelDist(refmodel, cosine):
    t2md = Topic2ModelDist(refmodel, cosine)
    print t2md.id
    return CachedFunction(t2md, function_cache_folder, saveEvery=1)

def cachedTopic2ModelMatch(refmodel, matcher):
    t2mm = Topic2ModelMatch(refmodel, matcher)
    print t2mm.id
    return CachedFunction(t2mm, function_cache_folder, saveEvery=1)

if __name__ == '__main__':
    print supervisedMatcherV1().id