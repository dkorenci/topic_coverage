from pytopia.context.ContextResolver import resolve

from pytopia.resource.builder_cache.ResourceBuilderCache import ResourceBuilderCache
from phenotype_context.phenotype_corpus.construct_corpus import CORPUS_ID as PHENO_CORPUS
USPOL_CORPUS = 'us_politics_textperline'
from gtar_context.semantic_topics.construct_model import MODEL_ID as GTAR_REFMODEL
from phenotype_context.phenotype_topics.construct_model import MODEL_DOCS_ID as PHENO_REFMODEL

from topic_coverage.experiments.measure_factory import supervisedModelCoverage, ctcModelCoverage
from topic_coverage.resources.pytopia_context import topicCoverageContext
from topic_coverage.modelbuild.modelbuild_docker_v1 import modelset, msetFilter
from pytopia.topic_model.TopicModel import TopicModel

from topic_coverage.settings import stability_mock_models, topic_models_extended

def loadGroupByT(modelfolder, corpus=None, modelType=None):
    if not isinstance(modelfolder, list): modelfolder = [modelfolder]
    mctx = None
    for mf in modelfolder:
        #ctx = ResourceBuilderCache.loadResources(mf, asContext=True)
        filter = msetFilter(modelType, None, corpus)
        ctx = modelset(mf, filter, ctx=True)
        if mctx is None: mctx = ctx
        else: mctx.merge(ctx)
    models = [m for m in mctx]
    #def corpFilter(m): return False if corpus and m.corpus != corpus else True
    #models = [m for m in models if corpFilter(m)]
    byT = {m.numTopics():[] for m in models}
    for m in models: byT[m.numTopics()].append(m)
    return byT, mctx

def loadGroupByTypeAndT(modelfolder, corpus=None, modelTypes=['lda', 'alda', 'nmf', 'pyp']):
    mctx = None; byType = {}
    for modelType in modelTypes:
        filter = msetFilter(modelType, None, corpus)
        ctx = modelset(modelfolder, filter, ctx=True)
        # group models by T
        models = [m for m in ctx]
        byT = {m.numTopics(): [] for m in models}
        for m in models: byT[m.numTopics()].append(m)
        byType[modelType] = byT
        # add models to context
        if mctx is None: mctx = ctx
        else: mctx.merge(ctx)
    return byType, mctx

def convertTopicModelsToMock(modelfolder, savefolder, corpus=None, modelType=None):
    filter = msetFilter(modelType, None, corpus)
    models = ResourceBuilderCache.loadResources(modelfolder, filter=filter, asContext=False, loadMock=TopicModel)
    mockCache = ResourceBuilderCache(None, savefolder)
    for m in models:
        print type(m), m.numTopics(), m.id
        mockCache._saveToDiskCache(m)


def plotCoverageByTMultiple(refmodel, models, covMeasure, fname=None, mtypes=None):
    '''
    Plot coverage of refmodel as a function of T, for a set of models,
    expectedly of the same type.
    :param models: map modeltype -> { map numTopics -> list of models }
    '''
    from matplotlib import pyplot as plt
    import numpy as np
    # prepare plotting
    fig, axes = plt.subplots(1, 1)
    cmap = plt.get_cmap('terrain')
    # prepare data
    refmodel = resolve(refmodel); numTagg = []
    if mtypes is None: mtypes = models.keys()
    numtypes = len(mtypes)
    for ti, mtype in enumerate(mtypes):
        modelsByT = models[mtype]
        numTs = sorted(modelsByT.keys())
        numTagg.extend(numTs)
        results = [[covMeasure(refmodel, m) for m in modelsByT[t]] for t in numTs ]
        # plot, scatter + boxplot per T
        #axes.boxplot(results, positions=numTs, showfliers=False)
        #xcoord = range(1, len(results)+1) # x coordinates of boxes
        xcoord = numTs
        #for i, res in enumerate(results):
            #axes.scatter([xcoord[i]] * len(res), res, alpha=0.4)
            # plot the average
        avg = np.average(results, axis=1)
        clr = cmap(ti / float(numtypes+3))
        axes.plot(numTs, avg, color=clr, marker='.', mew=1, markersize=5, label=mtype)
        #axes.xaxis.tick_top()
        # Set the labels
    numTagg = sorted(list(set(numTagg)))
    # labels = numTagg
    #axes.set_xticklabels(labels, minor=False)
    #for tick in axes.xaxis.get_major_ticks(): tick.label.set_fontsize(6)
    #for tick in axes.yaxis.get_major_ticks(): tick.label.set_fontsize(6)
    #plt.xticks(rotation=45)
    axes.yaxis.grid(True, linestyle='--')
    axes.xaxis.grid(True, linestyle='--')
    # Turn off x ticks
    for t in axes.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    axes.legend(loc='lower right', prop={'size': 7})
    #plt.tight_layout(pad=0)
    fname = 'cov4numt.pdf' if not fname else fname+'.pdf'
    fig.savefig(fname)
    #plt.show()

def plotCoverageByTSingle(refmodel, modelsByT, covMeasure, fname=None):
    '''
    Plot coverage of refmodel as a function of T, for a set of models,
    expectedly of the same type.
    :param modelsByT: map numTopics -> list of models
    '''
    from matplotlib import pyplot as plt
    import numpy as np
    # prepare data
    refmodel = resolve(refmodel)
    numTs = sorted(modelsByT.keys())
    results = [[covMeasure(refmodel, m) for m in modelsByT[t]] for t in numTs ]
    # plot, scatter + boxplot per T
    print 'COVERAGE: ', covMeasure.id
    fig, axes = plt.subplots(1, 1)
    axes.boxplot(results, positions=numTs, showfliers=False)
    #xcoord = range(1, len(results)+1) # x coordinates of boxes
    xcoord = numTs
    for i, res in enumerate(results):
        axes.scatter([xcoord[i]] * len(res), res, alpha=0.4)
        # plot the average
        avg = np.average(res)
        axes.plot(xcoord[i], avg, color='r', marker='x', mew=1, markersize=8)
    #axes.xaxis.tick_top()
    # Set the labels
    labels = numTs
    axes.set_xticklabels(labels, minor=False)
    for tick in axes.xaxis.get_major_ticks(): tick.label.set_fontsize(6)
    for tick in axes.yaxis.get_major_ticks(): tick.label.set_fontsize(6)
    #plt.xticks(rotation=45)
    axes.yaxis.grid(True, linestyle='--')
    axes.xaxis.grid(True, linestyle='--')
    # Turn off x ticks
    for t in axes.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    plt.tight_layout(pad=0)
    fname = 'cov4numt.pdf' if not fname else fname+'.pdf'
    fig.savefig(fname)
    #plt.show()

def plotGraphNew(): # plotting on large model sets by loading only subset at a time
    # get custom-cache coverage functions (extract factory methods)
    # load models (!memory - no perm. storage, with gc) and record numTs
    #     alt. do this in ad-hoc manner and store numTs to send as param
    # as in cacl_coverage, load model subset for each numT
    #  calculate coverages, store values for each cov-func
    # plot graph - receives list of numTs and for each cov. vals for each cov. func
    pass

def covXnumtopics(corpus='uspol', cov='ctc', mtype='lda',
                  mfolder='/data/modelbuild/topic_coverage/numt_test_nmf/',
                  byType=False, modelTypes=['lda', 'alda', 'nmf', 'pyp']):
    # create plot fname id
    typelab = '_'.join(sorted(modelTypes)) if byType else mtype
    fname = 'cov4numt_corpus[%s]_cov[%s]_model[%s]' % (corpus, cov, typelab)
    # setup corpus and refmodel ids
    if corpus == 'uspol':corpusId, refmodel = USPOL_CORPUS, GTAR_REFMODEL
    elif corpus == 'pheno':corpusId, refmodel = PHENO_CORPUS, PHENO_REFMODEL
    # construct coverage function
    from topic_coverage.experiments.modelparams.calc_coverage import constructCoverage
    if cov == 'ctc': cov = constructCoverage('ctc')
    elif cov == 'sup.strict' or cov == 'sup.ns':
        cov = constructCoverage(cov, corpus)
    # load models
    if not byType: groupedModels, mctx = loadGroupByT(mfolder, corpusId, mtype)
    else: groupedModels, mctx = loadGroupByTypeAndT(mfolder, corpusId, modelTypes)
    # create plot
    with mctx:
        if not byType: plotCoverageByTSingle(refmodel, groupedModels, cov, fname)
        else: plotCoverageByTMultiple(refmodel, groupedModels, cov, fname, modelTypes)

if __name__ == '__main__':
    with topicCoverageContext():
        #covXnumtopics('uspol', 'ctc')
        #covXnumtopics('pheno', 'sup.ns')
        #covXnumtopics('pheno', 'sup.strict', '/data/modelbuild/topic_coverage/numt_test_nmf2/')
        #covXnumtopics('pheno', 'ctc', 'pyp', '/data/resources/coverage_experiments/modelbuild/numt_prodbuild/')
        # '/data/modelbuild/topic_coverage/docker_modelbuild/numt_prodbuild_sample_small/'
        convertTopicModelsToMock(topic_models_extended, stability_mock_models, None, None)
