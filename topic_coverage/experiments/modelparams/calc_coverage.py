'''
Methods for calculacting cached coverage functions on large model sets,
by dividing the set into subsets that can fit into memory.
'''
from unittest.case import _AssertRaisesContext

from topic_coverage.modelbuild.modelset_loading import modelset1Families, modelsetLoad
from topic_coverage.experiments.coverage.experiment_runner import evaluateCoverage
from topic_coverage.experiments.measure_factory import \
        supervisedModelCoverage, supervisedTopicMatcher, ctcModelCoverage
from topic_coverage.resources.pytopia_context import topicCoverageContext
from gtar_context.semantic_topics.construct_model import MODEL_ID as GTAR_REFMODEL
from phenotype_context.phenotype_topics.construct_model import MODEL_DOCS_ID as PHENO_REFMODEL
from pytopia.context.ContextResolver import resolve

from gc import collect
from os.path import join

#functionCacheFolder = '/datafast/topic_coverage/numt_prod/function_cache/'
#functionCacheFolder = '/datafast/topic_coverage/numt_test/function_cache/'
functionCacheFolder = '/datafast/topic_coverage/numt_prod/function_cache_djurdja_working/'
supmodelcovCache = join(functionCacheFolder, 'sup_model_coverage')
ctcmodelcovCache = join(functionCacheFolder, 'ctc_model_coverage')
topicmatchCache = join(functionCacheFolder, 'topic_match')

def constructCoverage(type='sup.strict', corpus='uspol',
                      supCovCache=supmodelcovCache, ctcCovCache=ctcmodelcovCache,
                      topicMatchCache=False):
    ''' Factory method for coverage functions. '''
    if type == 'ctc': cov = ctcModelCoverage(strict=False, cached=ctcCovCache)
    elif type == 'sup.strict' or type == 'sup.ns':
        strict = (type == 'sup.strict')
        cov = supervisedModelCoverage(corpus, strict,
                        covCache=supCovCache, matchCache=topicMatchCache)
    return cov

def calcCoverage(folders, corpus, modelFamilies, numT, covTyp='ctc',
                 supCovCache=supmodelcovCache, ctcCovCache=ctcmodelcovCache, topicMatchCache=False):
    print "calcCoverage START"
    if not isinstance(folders, list): folders = [folders]
    for f in folders:
        modelsets, modelCtx, labels = \
            modelsetLoad(corpus=corpus, modelsFolder=f, families=modelFamilies, numT = numT)
        collect()
        print "modelset loaded"
        covFunc = constructCoverage(covTyp, corpus, supCovCache=supCovCache,
                                    ctcCovCache=ctcCovCache, topicMatchCache=topicMatchCache)
        print "coverage constructed"
        if corpus == 'uspol': refmodel = resolve(GTAR_REFMODEL)
        elif corpus == 'pheno': refmodel = resolve(PHENO_REFMODEL)
        for models in modelsets:
                print 'NUM models:', len(models)
                print ' topics: ', ','.join('%d' % m.numTopics() for m in models)
        with modelCtx:
            for models in modelsets:
                for m in models:
                    print 'calculating coverage for model: %s' % m.id
                    covFunc(refmodel, m)
    print 'calcCoverage FINISHED.'

def testRun1():
    #TODO if sup.cov cache size is still a problem, run manually for more topic size subsets
    #    or implement shell scripting with params: numT, corpus, modeltype
    f1, t1 = '/data/modelbuild/topic_coverage/numt_test_nmf/', range(20, 201, 20)
    f2, t2 = '/data/modelbuild/topic_coverage/numt_test_nmf2/', range(225, 401, 25)
    f, top = f1, t1
    for t in top:
        calcCoverage(f, 'pheno', numT = [t], modelFamilies=['nmf'])
        print 'T=%d finished' % t

def datasetStatistics(corpus='uspol', mfolder='/data/modelbuild/topic_coverage/docker_modelbuild/numt_prodbuild/'):
    from topic_coverage.modelbuild.modelbuild_docker_v1 import modelset, msetFilter
    if corpus == 'uspol': corpus = 'us_politics_textperline'
    elif corpus == 'pheno': corpus = 'pheno_corpus1'
    numModels = 0
    print 'CORPUS: %s' % corpus.upper()
    for mtyp in ['lda', 'nmf', 'pyp']: # 'alda',
        print 'TYPE: %s' % mtyp.upper()
        filter = msetFilter(mtyp, None, corpus)
        mset = modelset(mfolder, filter)
        numModelsTyp = len(mset)
        topics = [m.numTopics() for m in mset]
        print '#models: %d' % numModelsTyp
        print 'topics: %s' % (','.join('%d'%t for t in sorted(topics)))
        numModels += numModelsTyp
    print 'TOTAL #models %d' % numModelsTyp

# TODO unneccesary, move code to calcCovProdTest
def calcCovProd(topicSets, covType, numModels, corpus='uspol', modelFamilies='all',
                modelFolder='/data/modelbuild/topic_coverage/docker_modelbuild/numt_prodbuild/'):
    # TODO if sup.cov cache size is still a problem, run manually for more topic size subsets
    #    or implement shell scripting with params: numT, corpus, modeltype
    for topicSet in topicSets:
        print 'T=%s started' % topicSet
        calcCoverage(modelFolder, corpus, numT = topicSet, modelFamilies=modelFamilies,
                     covTyp=covType, numModels=numModels)
        print 'T=%s finished' % topicSet

def calcCovProdTest(rseed=8391776):
    from random import seed, shuffle
    alltopics = range(20, 501, 20)
    seed(rseed); shuffle(alltopics)
    chunkSize = 2
    topicChunks = [alltopics[i:i+chunkSize] for i in range(0, len(alltopics), chunkSize)]
    calcCovProd(topicChunks, 'ctc', 5, 'pheno', ['alda'])

def checkModelbuild(modelFolder='/data/modelbuild/topic_coverage/docker_modelbuild/numt_prodbuild/'):
    from pytopia.resource.builder_cache.ResourceBuilderCache import ResourceBuilderCache
    ResourceBuilderCache.loadResources(modelFolder, asContext=False);
        # diag code to insert into loadResources code
        # print r.id
        # numt = r.numTopics()
        # mtx = r.topicMatrix()
        # numUnsavedRes += 1
        # # res.append(r)

if __name__ == '__main__':
    with topicCoverageContext():
        #testRun1()
        #datasetStatistics('pheno')
        #calcCovProdTest()
        checkModelbuild()