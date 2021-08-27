from os import path

from pyutils.file_utils.location import FolderLocation as loc
from phenotype_context.dictionary.create_4outof5_dictionary import DICT_ID
from phenotype_context.phenotype_corpus.construct_corpus import CORPUS_ID as PHENO_CORPUS_ID
from pytopia.measure.topic_distance import *
from pytopia.resource.builder_cache.ResourceBuilderCache import ResourceBuilderCache
from pytopia.tools.parameters import flattenParams as fp, joinParams as jp
from topic_coverage.settings import resource_folder
from topic_coverage.topicmatch.distance_sampling import intervals

modelfolder = loc(path.join(resource_folder, 'test_models'))

import random, numpy as np
import copy

# base paramsets with build resources
uspolBase = { 'corpus':'us_politics_textperline', 'dictionary':'us_politics_dict', 'text2tokens':'whitespace_tokenizer' }
phenoBase = { 'corpus':PHENO_CORPUS_ID, 'dictionary':DICT_ID, 'text2tokens':'whitespace_tokenizer' }
#
hcaBase = { 'hcaLocation': '/data/code/hca/HCA-0.63/hca/hca', 'threads': 1, }

def addBase(params, base, hca=True, flatten=False):
    if flatten: params = fp(params)
    if hca: params = jp(params, [hcaBase])
    return jp(params, [base])

def unfoldSeed(rseed, rnge=(0, 5000000)):
    '''
    :param rseed: pair (numSeeds, startSeed)
    :param rnge: pair (min, max)
    :return: list of numSeeds random seeds, sampled from range
    '''
    num, start = rseed; mn, mx = rnge
    random.seed(start)
    return [random.randint(mn, mx) for i in range(num)]

def iterForModelHca(type, T, **param):
    '''
    Return number of gibbs iterations for a combination of a model and a number of topics.
    :param type: HcaAdapter model type
    :param T: number of topics
    :return: map of parameters determining number of iterations
    '''
    # check is the models is to be 'random' ie. unconverged,
    # with params constructed by initialization process
    if 'rndmodel' in param and param['rndmodel']: rndmodel = True
    else: rndmodel = False
    # set number of iterations depending on the model
    if type in ['lda', 'lda-asym']: # brn is used only for hdp
        C, brn, Cme, Bme = 500, 50, 200, 50
    elif type in ['pyp-doctop', 'pyp', 'hdp']:
        C, brn, Cme, Bme = 1000, 50, 400, 50
    if not rndmodel: return {'C':C, 'burnin':brn, 'Cme':Cme, 'Bme':Bme}
    else: return {'C':15, 'burnin':10, 'Cme':15, 'Bme':10} #return {'C':1, 'burnin':1, 'Cme':5, 'Bme':5}

def generateSeeds(model, corpus, T, numSeeds, rndSeed=None, rnge=(0, 100000000)):
    '''
    Create a sequence of numSeeds numbers randomly chosen from specified range.
    Radnom seed is deterministically derived from (model, corpus, T, rndSeed).
    '''
    sig = '%s_%s_%d' % (model, corpus, T)
    initSeed = long(hash(sig))
    if rndSeed: initSeed += rndSeed
    random.seed(initSeed)
    mn, mx = rnge
    return [random.randint(mn, mx) for i in range(numSeeds)]

def seedVariants(model, corpus, topics, numSeeds,
                 rseed=None, topicPar='T', seedPar='rseed', **params):
    '''
    Create sets of parameters for model building, each containing
    number of topics and random seed parameters.
    :param numSeeds: number of param. sets generated
    :param topicPar, seedPar: parameter names
    :param params: any addition parameters to be added to each param. set
    '''
    opts = []
    for T in topics:
        seeds = generateSeeds(model, corpus, T, numSeeds, rseed)
        for i in range(numSeeds):
            p = { topicPar: T, seedPar: seeds[i] }
            p.update(params)
            opts.append(p)
    return opts

def createParamset(corpus, model, numModels, topics, rseed=None, rndmodel=False):
    '''
    :param corpus: 'uspol' or 'pheno'
    :param model: 'nmf' or one of HcaAdapter model types
    :param topics: a single number or a list of topics
    :param rseed: (number of models with diff. rnd. seed per parameter set, initial seed)
    :return:
    '''
    if not isinstance(topics, list): topics = [topics]
    if corpus == 'uspol': base = uspolBase
    elif corpus == 'pheno': base = phenoBase
    if model == 'nmf':
        modelopts = seedVariants(model, corpus, topics, numModels, rseed, seedPar='rndSeed')
        modelopts = addBase(modelopts, base, hca=False)
    elif model in ['lda', 'lda-asym', 'hdp', 'pyp-doctop', 'pyp']:
        modelopts = seedVariants(model, corpus, topics, numModels, rseed, type=model)
        for p in modelopts:
            p.update(iterForModelHca(rndmodel=rndmodel, **p))
        modelopts = addBase(modelopts, base, hca=True)
    return modelopts

def paramset1(split, numSplits, numModels=10, rseed=None,):
    pset = []
    for corpus in ['uspol', 'pheno']:
        for model in ['nmf', 'lda', 'lda-asym']:
            pset.extend(createParamset(corpus, model, numModels,
                                       topics=[50, 100, 200], rseed=rseed))
        for model in ['hdp', 'pyp-doctop', 'pyp']:
            pset.extend(createParamset(corpus, model, numModels, topics=300, rseed=rseed))
    if split: return shuffleAndSplit(pset, rseed, numSplits)
    else: return pset

def paramset_lab(corpus, split, numSplits, numModels=10, rseed=None):
    '''
    Paramset for generating models for pair labeling on
    either us politics corpus (corpus=='uspol') or phenotype corpus (corpus=='pheno').
    '''
    pset = []
    for model in ['nmf', 'lda', 'lda-asym']:
        pset.extend(createParamset(corpus, model, numModels,
                                   topics=[50, 100, 200], rseed=rseed))
    for model in ['pyp']:
        pset.extend(createParamset(corpus, model, numModels, topics=300, rseed=rseed))
    if split: return shuffleAndSplit(pset, rseed, numSplits, stratype=[])
    else: return pset

def paramset_prod(split, numSplits, numModels=10, rseed=None, rndmodel=False):
    '''
    Paramset for generating production models for final experiments.
    '''
    pset = []
    for corpus in ['uspol', 'pheno']:
        for model in ['nmf', 'lda', 'lda-asym']:
            pset.extend(createParamset(corpus, model, numModels,
                                       topics=[50, 100, 200], rseed=rseed, rndmodel=rndmodel))
        for model in ['pyp']:
            pset.extend(createParamset(corpus, model, numModels, topics=300, rseed=rseed, rndmodel=rndmodel))
    if split: return shuffleAndSplit(pset, rseed, numSplits, stratype=['lda-asym', 'pyp'])
    else: return pset

def paramsetValid1(split, numSplits, rseed):
    '''
    Paramset for modelset built to validate quality
    metrics for different number of train cycles.
    '''
    #500, 50, 200, 50
    paramIter = [
        {'C': 100, 'burnin': 10, 'Cme': 50, 'Bme': 10},
        {'C': 250, 'burnin': 20, 'Cme': 50, 'Bme': 20},
        {'C': 500, 'burnin': 50, 'Cme': 200, 'Bme': 50},
        {'C': 1000, 'burnin': 50, 'Cme': 400, 'Bme': 50},
    ]
    # 1000, 50, 200, 50
    nonparamIter = [
        #{'C': 100, 'burnin': 10, 'Cme': 50, 'Bme': 10},
        {'C': 250, 'burnin': 20, 'Cme': 100, 'Bme': 20},
        #{'C': 500, 'burnin': 50, 'Cme': 200, 'Bme': 50},
        {'C': 1000, 'burnin': 50, 'Cme': 400, 'Bme': 50},
    ]
    def addIterParams(params, paramlist):
        newparams = []
        for p in params:
            for i, ip in enumerate(paramlist):
                np = copy.copy(p)
                np.update(ip)
                np['rseed'] = p['rseed'] + i*i*37
                newparams.append(np)
        return newparams
    pset = []; initSeed = rseed; numModels = 3
    for corpus in ['uspol', 'pheno']:
        for model in ['lda', 'lda-asym']:
            params = createParamset(corpus, model, topics=[200], rseed=(numModels, initSeed))
            params = addIterParams(params, paramIter)
            pset.extend(params)
            initSeed += 1
        for model in ['pyp']:
            params = createParamset(corpus, model, topics=[200, 300], rseed=(numModels, initSeed))
            params = addIterParams(params, nonparamIter)
            pset.extend(params)
            initSeed += 1
    if split: return shuffleAndSplit(pset, initSeed, numSplits)
    else: return pset

def paramsetValid2(split, numSplits, rseed):
    '''
    Paramset for validation, creating models with larger number of topics and longer convergence.
    '''
    paramIter = [
        {'C': 1000, 'burnin': 50, 'Cme': 400, 'Bme': 50},
        {'C': 1500, 'burnin': 50, 'Cme': 400, 'Bme': 50},
    ]
    nonparamIter = [
        {'C': 1000, 'burnin': 50, 'Cme': 400, 'Bme': 50},
        {'C': 1500, 'burnin': 50, 'Cme': 400, 'Bme': 50},
    ]
    def addIterParams(params, paramlist):
        newparams = []
        for p in params:
            for i, ip in enumerate(paramlist):
                np = copy.copy(p)
                np.update(ip)
                np['rseed'] = p['rseed'] + i*i*37
                newparams.append(np)
        return newparams
    pset = []; initSeed = rseed; numModels = 2
    for corpus in ['uspol', 'pheno']:
        for model in ['lda', 'lda-asym']:
            params = createParamset(corpus, model, topics=[500], rseed=(numModels, initSeed))
            params = addIterParams(params, paramIter)
            pset.extend(params)
            initSeed += 1
        for model in ['pyp']:
            params = createParamset(corpus, model, topics=[500], rseed=(numModels, initSeed))
            params = addIterParams(params, nonparamIter)
            pset.extend(params)
            initSeed += 1
    if split:
        if len(pset) <= numSplits: return [ [p] for p in pset ]
        else: return shuffleAndSplit(pset, initSeed, numSplits)
    else: return pset


def paramsetTest1(split=False, numSplits=3):
    pset = []; initSeed = 32; numModels=3
    for corpus in ['uspol']:
        for model in ['lda-asym', 'nmf']:
            pset.extend(createParamset(corpus, model, topics=[50, 100], rseed=(numModels, initSeed)))
            initSeed += 1
    if split: return shuffleAndSplit(pset, initSeed, numSplits)
    else: return pset

def paramsetTest2(split=False, numSplits=3):
    pset = []; initSeed = 32; numModels=3
    for corpus in ['uspol', 'pheno']:
        for model in ['lda-asym', 'nmf']:
            pset.extend(createParamset(corpus, model, topics=[50, 100], rseed=(numModels, initSeed)))
            initSeed += 1
        for model in ['hdp', 'pyp']:
            pset.extend(createParamset(corpus, model, topics=200, rseed=(numModels, initSeed)))
            initSeed += 1
    if split: return shuffleAndSplit(pset, initSeed, numSplits)
    else: return pset

def shuffleAndSplit(params, rndseed, split, stratype=['lda-asym', 'pyp']):
    params = copy.copy(params)
    if rndseed is None: rndseed = 665128
    random.seed(rndseed)
    res = [None]*split
    for i in range(split): res[i] = []
    def splitAndDistribute(store, items):
        random.shuffle(items)
        ss = np.array_split(items, split)
        random.shuffle(ss)
        #print len(ss)
        #for s in ss: print s
        for i, s in enumerate(ss):
            store[i].extend(s)
    for t in stratype:
        sel = [p for p in params if 'type' in p and p['type'] == t]
        #print t
        #print sel
        splitAndDistribute(res, sel)
        #print
    rest = [p for p in params if 'type' not in p or p['type'] not in stratype]
    splitAndDistribute(res, rest)
    return res

def buildModels(params, buildFolder):
    '''
    :param params: list of model hyperparams
    :param buildFolder: folder for storing built models
    :return:
    '''
    from pytopia.adapt.scikit_learn.nmf.adapter import SklearnNmfBuilder
    from pytopia.adapt.hca.HcaAdapter import HcaAdapterBuilder
    from pyutils.logging_utils.setup import createLogger, INFO
    import sys
    hcaBuilder = ResourceBuilderCache(HcaAdapterBuilder, buildFolder)
    nmfBuilder = ResourceBuilderCache(SklearnNmfBuilder(), buildFolder)
    models = []
    log = createLogger(buildModels.__module__+'.'+buildModels.__name__, INFO)
    for p in params:
        try:
            if 'type' in p:
                m = hcaBuilder(**p)
            else: m = nmfBuilder(**p)
            models.append(m)
            print m.id
        except:
            from traceback import format_exception
            e = sys.exc_info()
            strace = ''.join(format_exception(e[0], e[1], e[2]))
            log.error('build for params failed: %s' % p)
            log.error('stacktrace:\n%s'%strace)
    return models

def plotBuildCoverage(cacheFolder, filters, labels, distance='cosine', corpus='us_politics', numIntervals=10):
    from topic_coverage.experiments.coverage.coverage_plots import coverageForThresholdsBars
    modelsets = [ResourceBuilderCache.loadResources(cacheFolder, filter=f, asContext=False)
                 for f in filters]
    for ms in modelsets:
        for m in ms: print m.id
        #for m in ms: print m
        print
    if corpus == 'us_politics':
        if distance == 'cosine':
            thresholds = intervals(0.0, 1.0, numIntervals, flat=True)[1:]
            distMetric = cosine
        elif 'kl' in distance.lower():
            thresholds = intervals(0.0, 7.0, numIntervals, flat=True)[1:]
            distMetric = klDivZero
    else:
        thresholds = intervals(0.0, 1.0, numIntervals, flat=True)[1:]
        distMetric = cosine
    print thresholds
    label = '%s_coverage_%s'%(corpus,'_'.join(labels))
    coverageForThresholdsBars(corpus, modelsets, thresholds, distMetric=distMetric,
                              labels=labels, plotlabel=label)

def plotBuildCoverage2(modelsets, labels, distance='cosine', corpus='us_politics', numIntervals=10):
    from topic_coverage.experiments.coverage.coverage_plots import coverageForThresholdsBars
    for ms in modelsets:
        for m in ms: print m.id
        print
    if corpus == 'us_politics':
        if distance == 'cosine':
            thresholds = intervals(0.0, 1.0, numIntervals, flat=True)[1:]
            distMetric = cosine
        elif 'kl' in distance.lower():
            thresholds = intervals(0.0, 7.0, numIntervals, flat=True)[1:]
            distMetric = klDivZero
    else:
        if distance == 'cosine':
            thresholds = intervals(0.0, 1.0, numIntervals, flat=True)[1:]
            distMetric = cosine
    print thresholds
    label = '%s_coverage_%s'%(corpus,'_'.join(labels))
    coverageForThresholdsBars(corpus, modelsets, thresholds, distMetric=distMetric,
                              labels=labels, plotlabel=label)


def plotParamsetTest1():
    plotBuildCoverage('/datafast/topic_coverage/docker_modelbuild/djurdja_paramsetTest1/',
                      [['T[50]', 'lda-asym'], ['T[100]', 'lda-asym'],
                       ['Nmf', 'T[50]'], ['Nmf', 'T[100]']],
                      ['ldasym50', 'ldasym100', 'nmf50', 'nmf100'])

def plotParamsetTest2(corpus='us_politics'):
    plotBuildCoverage('/datafast/topic_coverage/docker_modelbuild/djurdja_paramsetTest2/',
                      [['T[100]', corpus, 'lda-asym'], ['Nmf', 'T[100]', corpus],
                       [corpus, 'hdp'], [corpus, 'pyp']],
                       ['ldasym100', 'nmf100', 'hdp200', 'pyp200'], corpus=corpus)

def modelset(modelsFolder, filter, num=None, ctx=False):
    models = ResourceBuilderCache.loadResources(modelsFolder, filter=filter, asContext=ctx)
    if num: models = models[:num]
    return models

def msetFilter(family, T, corpus):
    '''
    :param family: 'lda', 'alda', 'pyp', 'nmf', 'artm'
    :param T: number of topics
    '''
    tlab = 'T[%d]'%T if T else 'T'
    if family == 'lda': return [tlab, corpus, 'lda]']
    elif family == 'alda': return [tlab, corpus, 'lda-asym']
    elif family == 'pyp': return [corpus, 'pyp]']
    elif family == 'nmf': return ['Nmf', tlab, corpus]
    elif family == 'artm': return ['Artm', tlab, corpus]

def msetlab(family, T): return '%s%d'%(family, T)

def plotBuild(paramset, numTopics=None, corpus='us_politics', dist='cosine', intervals=10,
              modelsFolder='/datafast/topic_coverage/docker_modelbuild/djurdja_build1/'):
    if paramset == 'nonparam':
        modelsetFilters = [[corpus, 'hdp'], [corpus, 'pyp-doctop'], [corpus, 'pyp]']]
        modelsetLabels = ['hdp300', 'pypdoc300', 'pyp300']
    elif paramset in ['param', 'lda', 'alda', 'nmf']:
        if not isinstance(numTopics, list): numTopics = [numTopics]
        modelsetFilters = []
        modelsetLabels = []
        for t in numTopics:
            top = 'T[%d]'%t
            msets, lsets = [], []
            if paramset in ['param', 'lda']:
                msets.extend([top, corpus, 'lda]'])
                lsets.extend('lda%d'%t)
            if paramset in ['param', 'alda']:
                msets.extend([top, corpus, 'lda-asym'])
                lsets.extend('ldasym%d'%t)
            if paramset in ['param', 'nmf']:
                msets.extend(['Nmf', top, corpus])
                lsets.extend('nmf%d'%t)
            modelsetFilters.extend(msets); modelsetLabels.extend(lsets)
    elif paramset == 'best':
        modelsetFilters = [[corpus, 'hdp'], [corpus, 'pyp-doctop'], [corpus, 'pyp]']]
        modelsetLabels = ['hdp300', 'pypdoc300', 'pyp300']
        for t in [200]:
            top = 'T[%d]'%t
            msets = [[top, corpus, 'lda]'], [top, corpus, 'lda-asym'], ['Nmf', top, corpus]]
            lsets = ['lda%d'%t, 'ldasym%d'%t, 'nmf%d'%t]
            modelsetFilters.extend(msets); modelsetLabels.extend(lsets)
    elif paramset == 'heterogeneous':
        modelsetFilters = [[corpus, 'hdp'], [corpus, 'pyp]']]
        modelsetLabels = ['hdp300', 'pyp300']
        for t in [200]:
            top = 'T[%d]'%t
            msets = [[top, corpus, 'lda]'], [top, corpus, 'lda-asym'], ['Nmf', top, corpus]]
            lsets = ['lda%d'%t, 'ldasym%d'%t, 'nmf%d'%t]
            modelsetFilters.extend(msets); modelsetLabels.extend(lsets)
    elif paramset == 'validbuild2':
        modelsetFilters = [['C[1000]', corpus, 'lda]'], ['C[1000]', corpus, 'lda-asym'],
                           ['C[1000]', corpus,  'pyp]']]
        modelsetLabels = ['lda500', 'alda500', 'pyp500']
    elif paramset == 'top':
        modelsetFilters = [['T[200]', corpus, 'lda]'], ['T[200]', corpus, 'lda-asym'],
                           [corpus, 'pyp'], ['Nmf', 'T[200]', corpus], ]
        modelsetLabels = ['lda200', 'alda200', 'pyp300', 'nmf200']
    plotBuildCoverage(modelsFolder, modelsetFilters, modelsetLabels,
                      corpus=corpus, numIntervals=intervals, distance=dist)



def loadBuild1Modelset(modelset, numTopics=None, corpus='us_politics', asContext=False):
    '''Load a set of models from 'djurdja build1' build. '''
    cacheFolder = '/datafast/topic_coverage/docker_modelbuild/djurdja_build1/'
    topStr = 'T[%d]' % numTopics
    if modelset == 'lda': filter = [topStr, corpus, 'lda]']
    elif modelset == 'alda': filter = [topStr, corpus, 'lda-asym']
    elif modelset == 'pyp': filter = [corpus, 'pyp]']
    elif modelset == 'nmf': filter = ['Nmf', topStr, corpus]
    elif modelset == None: filter = None
    return ResourceBuilderCache.loadResources(cacheFolder, filter, asContext=asContext)

def pp(paramset):
    print len(paramset)
    for p in paramset:
        print p

def psplit(split):
    print 'PARTS', len(split)
    for i in range(len(split)):
        pp(split[i])
        print

def validSplit(paramset, numSplits, seed=5785):
    nosplit = paramset(False, rseed=seed)
    nosplit = set([ str(e) for e in nosplit ])
    split = paramset(True, numSplits, seed)
    split = set([ str(e) for s in split for e in s ])
    assert set(nosplit) == set(split)

def validParamsetIncrementality():
    '''
      Check that random seed are generated based on modeltype, corpus and numTopics,
      so paramsets with larger number of models per above combination
      must contain smaller paramsets.
    '''
    for seed in [None, 1, 88167]:
        pset = None
        for numVariants in [1, 5, 10]:
            newpset = paramset_lab(False, 0, numVariants, seed)
            if pset is not None:
                for p in pset:
                    assert p in newpset
            pset = newpset

def validBuild(cacheFolder, fullmodels=True, docs=False, mir=None):
    models = ResourceBuilderCache.loadResources(cacheFolder, asContext=False)
    print len(models)
    for m in models: print m.id
    if fullmodels:
        with ResourceBuilderCache.loadResources(cacheFolder, asContext=True):
            if not docs:
                for m in models:
                    print unicode(m)
                    print
            else:
                for i, m in enumerate(models):
                    if mir and i not in range(mir[0], mir[1]): continue
                    print m.id
                    for t in m.topicIds():
                        print ' '.join(m.topTopicWords(t, 20))
                        for t in m.topTopicDocs(t, 20): print '    ', t
                        print
                    print
                    print

    else:
        for m in models:
            print m.id

if __name__ == '__main__':
    #pp(paramsetTest2())
    #plotParamsetTest2('pheno_corpus1')
    #plotBuild1('us_politics')
    #plotBuild1('pheno_corpus1')
    # plotBuild('validbuild2', [50, 100, 200], 'us_politics', 'cosine', 20,
    #           modelsFolder='/datafast/topic_coverage/docker_modelbuild/djurdja_validbuild2/')
    # plotBuild('top', [50, 100], 'us_politics', 'cosine', 10,
    #           modelsFolder='/datafast/topic_coverage/docker_modelbuild/paramset_lab_uspol/')
        #plotBuild1('heterogeneous', [200], 'us_politics', 'kl', 10)
    #plotBuild1('best', None, 'pheno_corpus1', 20)
    #plotParamsetTest1()
    #pp(paramsetValid2(False, 24, 6571))
    #psplit(paramsetValid2(True, 24, 1234))
    #pp(paramset_lab_uspol(False, 0, 2, 83610))
    #psplit(paramset_lab_uspol(True, 20, 2, 83610))
    #testParamsetIncrementality()
    #pp(paramset_lab('pheno', False, 20, 2, 918468))
    #psplit(paramset_lab('pheno', True, 20, 2, 918468))
    #validBuild('/datafast/topic_coverage/docker_modelbuild/paramset_lab_pheno/', docs=False)
    # plotBuild('top', [50, 100, 200], 'pheno_corpus1',
    #           modelsFolder='/datafast/topic_coverage/docker_modelbuild/paramset_lab_pheno/')
    #pp(paramset_prod(False, -1, numModels=2, rseed=8771203, rndmodel=True))
    psplit(paramset_prod(True, 20, numModels=10, rseed=8771203, rndmodel=False))