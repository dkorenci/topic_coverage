from os import path

from file_utils.location import FolderLocation as loc
from pytopia.adapt.hca.HcaAdapter import HcaAdapterBuilder
from pytopia.resource.builder_cache.ResourceBuilderCache import ResourceBuilderCache
from pytopia.tools.parameters import flattenParams as fp, joinParams as jp
from topic_coverage.settings import resource_folder

modelfolder = loc(path.join(resource_folder, 'test_models'))

import copy

uspolBase = { 'corpus':'us_politics_textperline', 'dictionary':'us_politics_dict', 'text2tokens':'whitespace_tokenizer' }
hcaBase = {
    'hcaLocation': '/data/code/hca/HCA-0.63/hca/hca',
    #'tmpFolder': '/datafast/topic_coverage/modelbuild/testhca/tmp/',
    'threads': 1,
}

def addHcaBase(params): return jp(jp([uspolBase], [hcaBase]), fp(params))

lda50paramsQuick = {'type':'lda', 'T':50, 'C':1, 'burnin':1, 'Cme':10, 'Bme':5, 'rseed':range(1)}
lda50paramsQuick = addHcaBase(lda50paramsQuick)
#iter=5; brnin=2; iterme=3; brninme=1 #tiny
#iter=100; brnin=10; iterme=50; brninme=20 #medium
iter=300; brnin=30; iterme=100; brninme=20 #large
lda50params1 = {'type':'lda', 'T':50, 'C':iter, 'burnin':brnin,
                'Cme':iterme, 'Bme':brninme, 'rseed':range(2)}
lda50params2 = copy.copy(lda50params1); lda50params3 = copy.copy(lda50params1)
lda50params2['rseed'] = range(2,4)
lda50params3['rseed'] = range(4,6)
lda50params1 = addHcaBase(lda50params1)
lda50params2 = addHcaBase(lda50params2)
lda50params3 = addHcaBase(lda50params3)

buildsets = {'lda50params1':lda50params1, 'lda50params2':lda50params2,
             'lda50params3': lda50params3}

def buildHcaModels(params, buildFolder):
    '''
    :param params: list of model hyperparams
    :param buildFolder: folder for storing built models
    :return:
    '''
    print buildFolder
    builder = ResourceBuilderCache(HcaAdapterBuilder, buildFolder)
    models = []
    for p in params:
        print p
        m = builder(**p)
        models.append(m)
    return models

def runBuild():
    import sys
    bset = sys.argv[1]
    bfolder = loc(sys.argv[2])()
    buildHcaModels(buildsets[bset], bfolder)

def printBuild(cacheFolder):
    print ResourceBuilderCache.loadResources(cacheFolder)

def plotBuildCoverage(cacheFolder):
    from topic_coverage.experiments.coverage.coverage_plots import coverageForThresholdsBars
    #models100c = ResourceBuilderCache.loadResources(cacheFolder, filter=r'.*C\[100\].*', asContext=False)
    #models5c = ResourceBuilderCache.loadResources(cacheFolder, filter=r'.*C\[5\].*', asContext=False)
    #modelsets = [models5c] #,models100c]
    #models300c = ResourceBuilderCache.loadResources(cacheFolder, filter=r'.*C\[300\].*', asContext=False)
    ldaAsym50 = ResourceBuilderCache.loadResources(cacheFolder,
                                                   filter=r'.*T\[50\].*us_politics.*lda-asym.*',
                                                   asContext=False)
    ldaAsym100 = ResourceBuilderCache.loadResources(cacheFolder,
                                                   filter=r'.*T\[100\].*us_politics.*lda-asym.*',
                                                   asContext=False)
    #modelsets = [models100c, models300c]
    modelsets = [ldaAsym50, ldaAsym100]
    for ms in modelsets:
        for m in ms: print m.id
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    coverageForThresholdsBars('us_politics_textperline', modelsets,
                              thresholds, labels=['ldaAsym50', 'ldaAsym100'],
                              plotlabel='us_politics_hcaLda_coverageForTrainCycles')

if __name__ == '__main__':
    plotBuildCoverage('/datafast/topic_coverage/docker_modelbuild/hcaLdaTest/')
    #runBuild()
    #printBuild('/datafast/topic_coverage/docker_modelbuild/hcaLdaTest/')
    #testBuildHca()
    #buildHcaModels(lda50params1, modelfolder('hcaLda'))
    #buildHcaModels(lda50paramsQuick, 'hcaLda')
