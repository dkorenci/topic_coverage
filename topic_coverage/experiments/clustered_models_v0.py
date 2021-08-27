from pytopia.measure.avg_nearest_dist import \
    AverageNearestDistance, printAndDetails, TopicCoverDist
from pytopia.measure.topic_distance import cosine as cosineDist
from pytopia.measure.topic_similarity import cosine as cosineSim

from pytopia.context.ContextResolver import resolve
from pytopia.resource.loadSave import loadResource

from pytopia.topic_cluster.clustered_topics_model import ClusteredTopicsModelBuilder
from pytopia.topic_cluster.affprop_sklearn import AffpropSklearnTopicCluster
from pytopia.topic_cluster.hac_scipy import HacScipyTopicCluster
from pytopia.topic_cluster.kmedoid import KmedoidTopicCluster

from file_utils.location import FolderLocation as loc
from topic_coverage.settings import resource_folder

from pytopia.tools.parameters import flattenParams as fp, joinParams as jp, IdList

from topic_coverage.resources import pytopia_context
from topic_coverage.modelbuild.modelbuild_iter1 import modelsContext

from phenotype_context.phenotype_corpus.construct_corpus import CORPUS_ID as PHENO_CORPUS
from phenotype_context.dictionary.create_4outof5_dictionary import DICT_ID as PHENO_DICT
from phenotype_context.phenotype_topics.construct_model import MODEL_ID as PHENO_MODEL

def testAndExperiment():
    avgnd = AverageNearestDistance(cosineDist, pairwise=True)
    target = resolve('gtar_themes_model')
    #mfolder = loc('/datafast/topic_coverage/test_models/gensimLdaUsPoliticsParams_T[50]_initSeed[3245]_alpha[1.000]/')
    mfolder = loc('/datafast/topic_coverage/test_models/nmfSkelarnUsPoliticsParams_T[50]_initSeed[5661]/')
    models = mfolder.subfolders()[:15]
    clusterer = AffpropSklearnTopicCluster(similarity=cosineSim, preference=0.7)
    #clusterer = HacScipyTopicCluster(distance=cosineDist, linkage='average', numClusters=100)
    #clusterer = KmedoidTopicCluster(distance=cosineDist, numClusters=100)
    source = ClusteredTopicsModelBuilder(
                topics=models, topicsId='test_gtar_lda', clusterer=clusterer, aggregate='center')
    #source = loadResource(models[1])
    #print source
    #print target
    printAndDetails(target, source, avgnd)

def clusteringModelSet(id, clusterer, modelsFolder, numModels, numClusteredModels, rseed=445,
                        aggregate='average'):
    clmodels = IdList(); clmodels.id = id
    for i in range(numClusteredModels):
        models = loc(modelsFolder).subfolders(shuffle=True, seed=rseed+i)[:numModels]
        models = [ loadResource(mf, cache=True) for mf in models ]
        clModel = ClusteredTopicsModelBuilder(topics=models, topicsId=id,
                                          clusterer=clusterer, aggregate=aggregate)
        clmodels.append(clModel)
    return clmodels

def baseModelSet(id, modelsFolder, numModels, rseed=823, returnIds=False):
    clmodels = IdList(); clmodels.id = id
    models = loc(modelsFolder).subfolders(shuffle=True, seed=rseed)[:numModels]
    models = [ loadResource(mf, cache=True) for mf in models ]
    if returnIds:
        models = [ m.id for m in models ]
    return models

modelFolders = {
    ('us_politics', 'lda', 50) : '/datafast/topic_coverage/test_models/gensimLdaUsPoliticsParams_T[50]_initSeed[3245]_alpha[1.000]/',
    ('us_politics', 'nmf', 50) : '/datafast/topic_coverage/test_models/nmfSkelarnUsPoliticsParams_T[50]_initSeed[5661]/',
    ('us_politics', 'lda', 100) : '/datafast/topic_coverage/test_models/gensimLdaUsPoliticsParams_T[100]_initSeed[998]_alpha[0.500]/',
    ('us_politics', 'nmf', 100) : '/datafast/topic_coverage/test_models/nmfSkelarnUsPoliticsParams_T[100]_initSeed[5661]/',
    (PHENO_CORPUS, 'lda', 50): '/datafast/topic_coverage/test_models/gensimLdaPhenotypeParams_T[50]_initSeed[3245]/',
    (PHENO_CORPUS, 'nmf', 50): '/datafast/topic_coverage/test_models/nmfSklearnPhenotypeParams_T[50]_initSeed[5261]/',
    (PHENO_CORPUS, 'lda', 100): '/datafast/topic_coverage/test_models/gensimLdaPhenotypeParams_T[100]_initSeed[3245]/',
    (PHENO_CORPUS, 'nmf', 100): '/datafast/topic_coverage/test_models/nmfSklearnPhenotypeParams_T[100]_initSeed[5261]/'
}

def clusteringModelsSets1(corpus='us_politics', basemodel='lda', topics=50,
                          numModels=5, numInstances=5, numClusters=150):
    clusterers = [
        AffpropSklearnTopicCluster(similarity=cosineSim, preference=0.9),
        HacScipyTopicCluster(distance=cosineDist, linkage='average', numClusters=numClusters),
        KmedoidTopicCluster(distance=cosineDist, numClusters=numClusters)
    ]
    modelsets = []
    for cl in clusterers:
        mset = clusteringModelSet('id', cl, modelFolders[corpus, basemodel, topics],
                                  numModels, numClusteredModels=numInstances)
        modelsets.append(mset)
    return modelsets

def baseModelsSets1(corpus='us_politics', topics=50, numModels=5):
    modelsets = []
    basemodels = ['lda', 'nmf']
    for bm in basemodels:
        mset = baseModelSet('id', modelFolders[corpus, bm, topics], numModels)
        modelsets.append(mset)
    return modelsets

def coverageScoringExperiment(target, models, metrics):
    '''
    Evaluate coverage of the target models.
    For each group of models statistics for every metric is displayed.
    :param target: target model containing topics to be covered
    :param models: list of lists of models
    :param metrics: list of coverage-scoring metrics
            returning a value for a (target, covering) pair of models
    :return:
    '''
    from stat_utils.utils import Stats
    for modelset in models:
        print modelset[0].id
        for metric in metrics:
            scores = [ metric(target, model) for model in modelset ]
            print metric.id
            print Stats(scores)
            print ', '.join('%g'%s for s in scores)

coverMetrics = [
    AverageNearestDistance(cosineDist, pairwise=False),
    TopicCoverDist(cosineDist, 0.4)
]

def coverageClusteringUsPol(topics=50):
    target = resolve('gtar_themes_model')
    print 'CLUSTERING'
    print 'LDA %d TOPICS' % topics
    coverageScoringExperiment(target, clusteringModelsSets1(basemodel='lda', topics=topics), coverMetrics)
    print 'NMF %d TOPICS' % topics
    coverageScoringExperiment(target, clusteringModelsSets1(basemodel='nmf', topics=topics), coverMetrics)

def coverageClusteringPhenotype(topics=50):
    target = resolve(PHENO_MODEL)
    print 'CLUSTERING'
    print 'LDA %d TOPICS' % topics
    coverageScoringExperiment(target,
                              clusteringModelsSets1(basemodel='lda',
                                                    corpus=PHENO_CORPUS,
                                                    topics=topics, numClusters=120), coverMetrics)
    print 'NMF %d TOPICS' % topics
    coverageScoringExperiment(target,
                              clusteringModelsSets1(basemodel='nmf',
                                                    corpus=PHENO_CORPUS,
                                                    topics=topics, numClusters=120), coverMetrics)

def coverageBasemodelsUsPol(topics=50):
    target = resolve('gtar_themes_model')
    print 'BASEMODELS %d TOPICS' % topics
    coverageScoringExperiment(target, baseModelsSets1(topics=topics), coverMetrics)

def coverageBasemodelsPhenotype(topics=50):
    target = resolve(PHENO_MODEL)
    print 'BASEMODELS %d TOPICS' % topics
    coverageScoringExperiment(target,
                              baseModelsSets1(topics=topics,
                                              corpus=PHENO_CORPUS),
                              coverMetrics)

if __name__ == '__main__':
    with modelsContext():
        #testAndExperiment()
        #coverageClusteringLdaUsPol(100)
        coverageBasemodelsUsPol(100)
        #coverageBasemodelsPhenotype(100)
        #coverageClusteringPhenotype(100)
