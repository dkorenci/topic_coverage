''' Tests and checks related to model building '''

from topic_coverage.modelbuild.modelbuild_docker_v1 import validBuild, msetFilter

def printLabelingBuildParams():
    from topic_coverage.modelbuild.modelbuild_docker_v1 import modelset
    from phenotype_context import phenotypeContex
    from gtar_context import gtarContext
    phenoLabelingBuild = '/datafast/topic_coverage/docker_modelbuild/paramset_lab_pheno/'
    uspolLabelingBuild = '/datafast/topic_coverage/docker_modelbuild/paramset_lab_uspol/'
    for m in modelset(uspolLabelingBuild, None):
        print m.id
        print m.corpus, m.dictionary, m.text2tokens
    print
    for m in modelset(phenoLabelingBuild, None):
        print m.id
        print m.corpus, m.dictionary, m.text2tokens

productionBuildTestFolder = '/datafast/topic_coverage/modelbuild/production_testbuild/'
def productionBuildTest():
    from topic_coverage.resources import pytopia_context
    from topic_coverage.modelbuild.modelbuild_docker_v1 import buildModels, paramset_prod
    params = paramset_prod(split=0, numSplits=-1, numModels=1, rseed=8771203, rndmodel=True)
    modelFolder = productionBuildTestFolder
    buildModels(params, modelFolder)
    print params

def remapCorpusIndices(source, target):
    '''
    For two CorpusIndex-compatible objects on the same corpus (same set of text ids),
    create a mapping (bijection), of indices for source to indices of target using
    :return: map int -> int
    '''
    assert len(source) == len(target)
    N = len(source)
    def ids(ci): return set(id_ for id_ in ci)
    assert ids(source) == ids(target)
    #print ids(source)
    map = { si:target.id2index(source[si]) for si in range(N) }
    assert sorted(map.keys()) == range(N)
    assert sorted(map.values()) == range(N)
    #print map
    return map

def remapUspolCorpusIndices():
    from pytopia.utils.load import loadResource
    ciold = '/data/bckp/topic_coverage/resource_builders_05122018_before_prod/corpus_index/hid21516886537182116931381241551551588371111174242180197132397416811414237149235152111359310711198494014730816323821832819359/res0'
    cinew = '/datafast/topic_coverage/resource_builders/corpus_index/hid21516886537182116931381241551551588371111174242180197132397416811414237149235152111359310711198494014730816323821832819359/res0'
    ciold = loadResource(ciold)
    cinew = loadResource(cinew)
    return remapCorpusIndices(ciold, cinew)

def reorderNdarrayRows(mtx, remap):
    import numpy as np
    N = len(remap.keys())
    reord = np.zeros(N, np.int64)
    for oi, ni in remap.iteritems(): reord[ni] = oi
    return np.copy(mtx[reord, :])

uspolLablmodelsRemappedFolder = \
    '/datafast/topic_coverage/docker_modelbuild/paramset_lab_uspol_remapped/'
def remapUspolLabmodelCorpusTopics():
    '''
    Remap document-topic matrices (reshuffle rows) of a set of models created
    with one corpus topic index to match another corpus topic index.
    :return:
    '''
    folder = uspolLablmodelsRemappedFolder
    from file_utils.location import FolderLocation as loc
    from pytopia.resource.loadSave import loadResource, saveResource
    from pytopia.resource.builder_cache.ResourceBuilderCache import ResourceBuilderCache
    remap = remapUspolCorpusIndices()
    for fi in loc(folder).subfolders():
        if ResourceBuilderCache._isHidFolder(fi):
            for fj in loc(fi).subfolders():
                if ResourceBuilderCache._isResFolder(fj):
                    model = loadResource(fj)
                    mid = model.id
                    print mid
                    #print model.__dict__.keys()
                    if mid.startswith('HcaAdapter'): doctopAttr = '_docTopic'
                    elif mid.startswith('SklearnNmfTmAdapter'):
                        doctopAttr = '_SklearnNmfTmAdapter__w'
                    mtx = getattr(model, doctopAttr)
                    rmtx = reorderNdarrayRows(mtx, remap)
                    #print type(mtx)
                    setattr(model, doctopAttr, rmtx)
                    #print model.__dict__.keys()
                    saveResource(model, fj)

def printFullModels(mfolder, modelIndexRange=None):
    validBuild(mfolder, docs=True, mir=modelIndexRange)

def checkTestDockerBuild():
    folder = '/datafast/topic_coverage/modelbuild/b52_test_modelbuild/'
    printFullModels(folder)

def printModel(m):
    from topic_coverage.topicmatch.pair_labeling import textLabel
    print m.id
    for t in m.topicIds():
        print ' '.join(m.topTopicWords(t, 20))
        for t in m.topTopicDocs(t, 20, titles=False): print textLabel(t)
        print
    print
    print

def checkProdModelBuild(corpus='uspol', numModels=5, families='all', numT = [50, 100, 200]):
    from topic_coverage.modelbuild.modelset_loading import modelset1Families
    folder = '/datafast/topic_coverage/docker_modelbuild/prodbuild'
    modelsets, modelCtx, labels = modelset1Families(corpus, numModels, folder, families, numT)
    print labels
    with modelCtx:
        for i, ms in enumerate(modelsets):
            print 'FAMILY', labels[i]
            for m in ms: printModel(m)

def checkProdModelBuildVariants():
    #checkProdModelBuild('uspol', numModels=2, families=['pyp'], numT=[300])
    checkProdModelBuild('pheno', numModels=1, families='all', numT=[50, 100, 200])

if __name__ == '__main__':
    #printLabelingBuildParams()
    #productionBuildTest()
    #remapUspolCorpusIndices()
    #remapUspolLabmodelCorpusTopics()
    #pass
    #testRemappedUspollabModels()
    #printFullModels(productionBuildTestFolder, (10,13))
    #checkTestDockerBuild()
    checkProdModelBuildVariants()

