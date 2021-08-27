from pytopia.measure.topic_distance import klDivZero
from topic_coverage.modelbuild.modelbuild_docker_v1 import *

import numpy as np
from matplotlib import pyplot as plt

def paramsetRandom(modeltype=['pyp'], split=False, numSplits=3):
    pset = []; initSeed = 32; numModels=5
    for corpus in ['uspol', 'pheno']:
        for model in modeltype:
            pset.extend(createParamset(corpus, model, topics=[300],
                                       rseed=(numModels, initSeed), rndmodel=True))
            initSeed += 1
    if split: return shuffleAndSplit(pset, initSeed, numSplits)
    else: return pset

def buildRandomPyp():
    buildModels(paramsetRandom(), '/datafast/topic_coverage/modelbuild/randomhca/')

def buildTestLogRandom(modeltype=['pyp'], tmpfolder='/datafast/topic_coverage/modelbuild/hcatmp/'):
    ''' For inspecting the structure of hca log file. '''
    params = paramsetRandom(modeltype)[:1]
    if tmpfolder:
        for p in params: p['tmpFolder'] = tmpfolder
    buildModels(params, '/datafast/topic_coverage/modelbuild/randomhca_test/')

def distanceFromUniform(models, dist=cosine, label='unconverged'):
    # setup
    if not isinstance(models, list): models = [models]
    M = len(models); m = models[0]; W = len(m[0].vector)
    print m.id
    print W
    u = np.ones(W, np.float64) / W
    # calc distances from uniform and plot
    fig, ax = plt.subplots(M, 2)
    histParams = {'bins': 100, 'density': True}
    for i in range(M):
        model = models[i]
        vals = [dist(t.vector, u) for t in model]
        if M == 1: ax1, ax2 = ax[0], ax[1]
        else: ax1, ax2 = ax[i, 0], ax[i, 1]
        fsize=5
        for tick in ax1.get_xticklabels(minor=False): tick.set_fontsize(fsize)
        for tick in ax2.get_xticklabels(minor=False): tick.set_fontsize(fsize)
        for tick in ax1.get_yticklabels(minor=False): tick.set_fontsize(fsize)
        for tick in ax2.get_yticklabels(minor=False): tick.set_fontsize(fsize)
        ax1.yaxis.grid(True); ax2.yaxis.grid(True)
        ax1.boxplot(vals)
        ax2.hist(vals, **histParams)
    figid='distance_from_uniform_pyp_dist[%s]_%s'%(dist.__name__, label)
    plt.tight_layout(pad=0)
    plt.savefig(figid + '.pdf')

def loadPypModels(context=False, corpus='us_politics', unconverged=False):
    from pytopia.resource.builder_cache.ResourceBuilderCache import ResourceBuilderCache
    if unconverged: cacheFolder = '/datafast/topic_coverage/modelbuild/randomhca/'
    else: cacheFolder = '/datafast/topic_coverage/docker_modelbuild/djurdja_build1/'
    #corpus = 'pheno_corpus1'
    msets = [[corpus, 'pyp]']]
    models = [m for f in msets for m in
              ResourceBuilderCache.loadResources(cacheFolder, filter=f, asContext=False)]
    if context:
        from pytopia.context.Context import Context
        ctx = Context('iter0PhenoModelsContext')
        for m in models: ctx.add(m)
        return ctx
    else: return models

def distancePlots():
    for dist in [hellinger]: #[cosine, klDivZero, jensenShannon]:
        distanceFromUniform(loadPypModels(unconverged=False)[:5], dist=dist, label='converged')
        distanceFromUniform(loadPypModels(unconverged=True)[:5], dist=dist, label='unconverged')

if __name__ == '__main__':
    #distancePlots()
    #for p in paramsetRandomPyp(): print p
    #buildRandomPyp()
    buildTestLogRandom(tmpfolder=None)
    #buildTestLogRandom(modeltype=['lda'], tmpfolder=None)
                       #tmpfolder='/datafast/topic_coverage/modelbuild/hcatmplda/')