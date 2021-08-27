from topic_coverage.resources import pytopia_context

from topic_coverage.topicmatch.distance_sampling import *
from pytopia.measure.topic_distance import cosine
from topic_coverage.settings import resource_folder
from pyutils.file_utils.location import FolderLocation as loc

from phenotype_context.phenotype_corpus.construct_corpus import CORPUS_ID as PHENO_CORPUS
from phenotype_context.phenotype_topics.construct_model import MODEL_ID as PHENO_REFMODEL, \
        MODEL_DOCS_ID as PHENO_REFMODEL_DOCS
from phenotype_context.compose_context import phenotypeContex
from pytopia.context.ContextResolver import resolve

from topic_coverage.topicmatch.data_iter0 import loadDataset, labelProportions
from topic_coverage.topicmatch.data_analysis_iter0 import valueDist

import random, os, codecs
from textwrap import wrap

from topic_coverage.settings import topicpairs_sample_models

def phenoModelsLabeling(downsampleParam=False, downsampleNonparam=False,
                        refmodel=False, context=False, rseed=912):
    ''' Models for labeling to test and develop ternary scheme. '''
    from pytopia.resource.builder_cache.ResourceBuilderCache import ResourceBuilderCache
    cacheFolder = loc(topicpairs_sample_models)('pheno_models')
    corpus = PHENO_CORPUS
    paramTopics = [50, 100, 200]; nonparamTopics = [300]
    modelsetFilters = []
    ldaTag, nmfTag, ldasymTag, pypTag = 'lda]', 'Nmf', 'lda-asym]', 'pyp]'
    paramTags = [ldaTag, nmfTag, ldasymTag]
    for t in paramTopics:
        top = 'T[%d]'%t
        msets = [[top, corpus, ldaTag], [nmfTag, top, corpus], [top, corpus, ldasymTag]]
        modelsetFilters.extend(msets)
    for t in nonparamTopics:
        top = 'T[%d]'%t
        msets = [[top, corpus, pypTag], ]
        modelsetFilters.extend(msets)
    allmodels = []; random.seed(rseed)
    for f in modelsetFilters:
        models = ResourceBuilderCache.loadResources(cacheFolder, filter=f, asContext=False)
        dsample = None
        if downsampleParam and (sum(t in f for t in paramTags)): dsample = downsampleParam
        if downsampleNonparam and (pypTag in f): dsample = downsampleNonparam
        if dsample: models = random.sample(models, dsample)
        allmodels.extend(models)
    if refmodel:
        with phenotypeContex():
            allmodels.extend([resolve(PHENO_REFMODEL_DOCS)]*refmodel)
    if context:
        from pytopia.context.Context import Context
        ctx = Context('iter0PhenoModelsContext')
        for m in allmodels: ctx.add(m)
        return ctx
    else: return allmodels


def phenoRefmodels(instances=1, context=False):
    if context:
        from pytopia.context.Context import Context
        ctx = Context('phenoRefmodelContext')
        ctx.add(resolve(PHENO_REFMODEL_DOCS))
        return ctx
    else: return [resolve(PHENO_REFMODEL_DOCS)] * instances

def generatePhenoSchemedevelSet(dwnsmpPar=None, dwnsmpNpar=None, refmodel=None,
                                action='create_pairs', rseed=89431, stats='all'):
    ''' Topic pairs to test and develop ternary scheme.
    :stats: 'all', 'family', 'dist'
    '''
    pairFileId = 'phenoLabelingSchemedevel[%s,%s,%s]' % (dwnsmpPar, dwnsmpNpar, refmodel)
    sampleId = 'phenoLabelingSchemedevel'
    if action == 'create_pairs':
        models = phenoModelsLabeling(dwnsmpPar, dwnsmpNpar, refmodel, rseed=rseed)
        topics = [t for m in models for t in m]
        createDistances(topics, cosine, None, pairFileId, verbose=True)
    elif action == 'pair_stats':
        from topic_coverage.topicmatch.pair_labeling import createLabelingFiles
        fname = distancesFname(cosine, None, pairFileId)
        if stats in ['all', 'dist']: distancesPerInterval(fname, intervals(0, 1, 10))
        if stats == 'all': print
        if stats in ['all', 'family']: modelFamiliesPerSample(fname, intervals(0, 1, 10), 50,
                                                              rndseed=rseed)
    elif action == 'create_labeling':
        from topic_coverage.topicmatch.pair_labeling import createLabelingFiles
        fname = distancesFname(cosine, None, pairFileId)
        lfolder = loc(resource_folder)('topicmatch', 'labeling_pheno_production')
        with phenoModelsLabeling(dwnsmpPar, dwnsmpNpar, refmodel, context=True):
            createLabelingFiles(lfolder, sampleId, fname,
                            intervals(0, 1, 10), 50, docs=True, filesize=50, rndseed=rseed)

unlabeledPairsFolder = '/datafast/topic_coverage/topicmatch/labeling_pheno_production/'

unlabeledFiles=[
    'topicPairs[0-50].txt', 'topicPairs[50-100].txt', 'topicPairs[100-150].txt', 'topicPairs[150-200].txt',
    'topicPairs[200-250].txt', 'topicPairs[250-300].txt', 'topicPairs[300-350].txt', 'topicPairs[350-400].txt',
    'topicPairs[400-450].txt', 'topicPairs[450-500].txt',
]

def validateDataset():
    with phenoModelsLabeling(1, 1, 3, context=True):
        for f in unlabeledFiles:
            data = loadDataset(unlabeledPairsFolder, [f], nonlabeled=True)
            valueDist(data, [cosine], savefile=('iter1valdist[%s]'%f))

def phenoLabelingModelsContex():
    '''
    Create and return pytopia context with phenotype models used for generating topic pairs for labeling.
    '''
    return phenoModelsLabeling(1, 1, 3, context=True)

def printModels():
    for m in phenoModelsLabeling(1, 1):
        print m.id

if __name__ == '__main__':
    #printModels()
    #generatePhenoSchemedevelSet(1, 1, 3, action='create_pairs', stats='all', rseed=912)
    #generatePhenoSchemedevelSet(1, 1, 3, action='pair_stats', stats='family', rseed=21)
    generatePhenoSchemedevelSet(1, 1, 3, action='create_labeling', stats='family', rseed=21)
    #validateDataset()
