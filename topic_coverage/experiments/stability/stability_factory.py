'''
Factory for stability functions.
'''

from topic_coverage.settings import function_cache_folder
from os.path import join, exists
from os import mkdir
stabilityCache = join(function_cache_folder, 'stability')

from pytopia.evaluation.stability.modelset import ModelsetStability
from pytopia.evaluation.stability.modelmatch_bipartite import ModelmatchBipartite, TopicmatchVectorsim
from pytopia.evaluation.stability.modelmatch_conceptset import ModelmatchRelConceptset
from pytopia.measure.topic_similarity import cosine as cosineSim
from pytopia.topic_functions.cached_function import CachedFunction

from topic_coverage.experiments.measure_factory import supervisedTopicMatcher

def bipartiteStability(type='word-cosine', cacheFolder=stabilityCache, separateCache=False,
                       returnModelmatch=False):
    '''
    :param type: 'word' or 'doc'
    '''
    if (cacheFolder is None): cacheFolder = stabilityCache
    if separateCache: # create/use separate cache folder, using separateCache as folder name
        cacheFolder = join(cacheFolder, separateCache)
        if not exists(cacheFolder): mkdir(cacheFolder)
    if type == 'word-cosine': mm = ModelmatchBipartite('word-cosine')
    else: mm = ModelmatchBipartite(TopicmatchVectorsim(cosineSim, vectors=type))
    mm = CachedFunction(mm, cacheFolder, saveEvery=3)
    if returnModelmatch: return mm
    else: return ModelsetStability(mm)

def uniteBipartStabilCaches(target, sources, type='word-cosine'):
    '''
    Unite function param -> value mappings contained in separate caches
    into a single cached function.
    :param target: target cache folder
    :param sources: list of source cache folders
    :return:
    '''
    target = bipartiteStability(type, target, False, returnModelmatch=True)
    sources = [ bipartiteStability(type, s, False, returnModelmatch=True) for s in sources ]
    CachedFunction.unite(target, sources)

def relConceptsetStability(refmodel, matcher, cacheFolder=stabilityCache):
    '''
    :param type: 'word' or 'doc'
    '''
    if (cacheFolder is None): cacheFolder=stabilityCache
    mm = ModelmatchRelConceptset(refmodel, matcher)
    mm = CachedFunction(mm, cacheFolder, saveEvery=5)
    return ModelsetStability(mm)

def ctcStability(cacheFolder=stabilityCache):
    ''' Cached ctc stability based on nonstrict cosine ctc measure '''
    from topic_coverage.experiments.measure_factory import ctcModelCoverage
    from topic_coverage.experiments.stability.ctc_stability import ModelmatchCtc
    if (cacheFolder is None): cacheFolder = stabilityCache
    mm = ModelmatchCtc(ctcModelCoverage(cached=False))
    mm = CachedFunction(mm, cacheFolder, saveEvery=10)
    return ModelsetStability(mm)
