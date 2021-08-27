from topic_coverage.resources import pytopia_context

from topic_coverage.topicmatch.distance_sampling import *
from pytopia.measure.topic_distance import cosine
from topic_coverage.settings import resource_folder
from pyutils.file_utils.location import FolderLocation as loc

from gtar_context.semantic_topics.construct_model import MODEL_ID as USPOL_REF_MODEL
from pytopia.context.ContextResolver import resolve

from topic_coverage.topicmatch.data_iter0 import loadDataset, labelProportions
from topic_coverage.topicmatch.data_analysis_iter0 import valueDist

import random, os, codecs
from textwrap import wrap

from topic_coverage.settings import topicpairs_sample_models

def uspolModelsLabeling(downsampleParam=False, downsampleNonparam=False,
                             refmodel=False, context=False, rseed=75832):
    from pytopia.resource.builder_cache.ResourceBuilderCache import ResourceBuilderCache
    cacheFolder = loc(topicpairs_sample_models)('uspol_models')
    corpus = 'us_politics'
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
    if refmodel: allmodels.extend([resolve(USPOL_REF_MODEL)]*refmodel)
    #for m in allmodels: print m.id
    if context:
        from pytopia.context.Context import Context
        ctx = Context('iter0PhenoModelsContext')
        for m in allmodels: ctx.add(m)
        return ctx
    else: return allmodels


def uspolRefmodels(instances=1, context=False):
    if context:
        from pytopia.context.Context import Context
        ctx = Context('uspolRefmodelContext')
        ctx.add(resolve(USPOL_REF_MODEL))
        return ctx
    else: return [resolve(USPOL_REF_MODEL)] * instances

def generateUspolFinalLabelingSet(dwnsmpPar=None, dwnsmpNpar=None, refmodel=None,
                                    action='create_pairs', rseed=89431, stats='all'):
    ''' Topic pairs to test and develop ternary scheme.
    :stats: 'all', 'family', 'dist'
    '''
    pairFileId = 'uspolLabelingFinal[%s,%s,%s]' % (dwnsmpPar, dwnsmpNpar, refmodel)
    sampleId = 'uspolLabelingFinal'
    if action == 'create_pairs':
        models = uspolModelsLabeling(dwnsmpPar, dwnsmpNpar, refmodel)
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
        lfolder = loc(resource_folder)('topicmatch', 'labeling_uspol_final')
        with uspolModelsLabeling(dwnsmpPar, dwnsmpNpar, refmodel, context=True):
            createLabelingFiles(lfolder, sampleId, fname,
                            intervals(0, 1, 10), 50, docs=True, filesize=50, rndseed=rseed)

def outputTextCorpus(corpus, folder):
    cnt = -1
    corpus = resolve(corpus)
    def wrap_text(text, charsPerLine=100):
        return '\n'.join(wrap(text.replace('\n', ' '), charsPerLine))
    for txto in corpus:
        fname = os.path.join(folder, '%s.txt'%txto.id)
        f = codecs.open(fname, 'w', 'utf-8')
        if txto.title is not None:
            f.write('TITLE: '+txto.title+'\n')
        f.write(wrap_text(txto.text, 100))
        #f.write(txto.text)
        f.close()
        cnt -= 1
        if cnt == 0: break

unlabeledPairsFolder = '/home/damir/Dropbox/projekti/doktorat/D1 eksplorativa/mjerenje pokrivenosti/supervised/' \
                 'oznaceni parovi/labeling_uspol_final/neoznaceni parovi/'
unlabeledFiles=[
    'topicPairs[0-50].txt', 'topicPairs[50-100].txt', 'topicPairs[100-150].txt', 'topicPairs[150-200].txt',
    'topicPairs[200-250].txt', 'topicPairs[250-300].txt', 'topicPairs[300-350].txt', 'topicPairs[350-400].txt',
    'topicPairs[400-450].txt', 'topicPairs[450-500].txt',
]

def validateDataset():
    # with iter1UspolModelsTernarny(1, 1, 0, context=True):
    #     data = loadDataset(labeledPairsFolder, labeledFiles, nonlabeled=True)
    # valueDist(data, [cosine])
    with uspolModelsLabeling(1, 1, 3, context=True):
        for f in unlabeledFiles:
            data = loadDataset(unlabeledPairsFolder, [f], nonlabeled=True)
            valueDist(data, [cosine], savefile=('iter1valdist[%s]'%f))

def validateDataset():
    # with iter1UspolModelsTernarny(1, 1, 0, context=True):
    #     data = loadDataset(labeledPairsFolder, labeledFiles, nonlabeled=True)
    # valueDist(data, [cosine])
    with uspolModelsLabeling(1, 1, 3, context=True):
        for f in unlabeledFiles:
            data = loadDataset(unlabeledPairsFolder, [f], nonlabeled=True)
            valueDist(data, [cosine], savefile=('iter1valdist[%s]'%f))

def uspolLabelingModelsContex():
    '''
    Create and return pytopia context with uspol models used for generating topic pairs for labeling.
    '''
    return uspolModelsLabeling(1, 1, 3, context=True)

if __name__ == '__main__':
    #generateUspolFinalLabelingSet(1, 1, 3, action='create_pairs', stats='all')
    #generateUspolFinalLabelingSet(1, 1, 3, action='pair_stats', stats='family', rseed=17)
    generateUspolFinalLabelingSet(1, 1, 3, action='create_labeling', rseed=17)
    #validateDataset()
