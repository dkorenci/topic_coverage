'''
Experiments with varying thresholds for similarity.
'''

from file_utils.location import FolderLocation as loc
from phenotype_context.phenotype_corpus.construct_corpus import CORPUS_ID as PHENO_CORPUS
from phenotype_context.phenotype_topics.construct_model import MODEL_ID as PHENO_MODEL
from pytopia.context.ContextResolver import resolve
from pytopia.measure.avg_nearest_dist import \
    AverageNearestDistance, printAndDetails, TopicCoverDist
from pytopia.measure.topic_distance import cosine as cosineDist
from pytopia.measure.topic_similarity import cosine as cosineSim
from pytopia.resource.loadSave import loadResource
from pytopia.tools.parameters import IdList
from pytopia.topic_cluster.affprop_sklearn import AffpropSklearnTopicCluster
from pytopia.topic_cluster.clustered_topics_model import ClusteredTopicsModelBuilder
from pytopia.topic_cluster.hac_scipy import HacScipyTopicCluster
from pytopia.topic_cluster.kmedoid import KmedoidTopicCluster
from stat_utils.utils import Stats
from topic_coverage.experiments.coverage.experiment_runner import coverageScoringExperiment
from topic_coverage.modelbuild.modelbuild_iter1 import modelsContext


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

def baseModelsSets1(corpus='us_politics', topics=50, numModels=5, basemodels = ['lda', 'nmf']):
    modelsets = []
    for bm in basemodels:
        mset = baseModelSet('id', modelFolders[corpus, bm, topics], numModels)
        modelsets.append(mset)
    return modelsets

coverMetrics = [TopicCoverDist(cosineDist, th) for th in [0.3, 0.4, 0.5, 0.6]]

# def coverageClusteringUsPol():
#     target = resolve('gtar_themes_model')
#     print 'CLUSTERING'
#     print 'LDA %d TOPICS' % topics
#     coverageScoringExperiment(target, clusteringModelsSets1(basemodel='lda', topics=topics), coverMetrics)
#     print 'NMF %d TOPICS' % topics
#     coverageScoringExperiment(target, clusteringModelsSets1(basemodel='nmf', topics=topics), coverMetrics)

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

def coverageBasemodelsUsPol():
    target = resolve('gtar_themes_model')
    for topics in [50, 100]:
        print 'BASEMODELS %d TOPICS' % topics
        coverageScoringExperiment(target, baseModelsSets1(topics=topics), coverMetrics)

def plotCoverages(corpus, models, numTop, thresholds, numModels=10):
    '''
    Plot coverages of a set of topic models for varying topic similarity thresholds.
    Each model is defined as a combination of a model and a number of topics.
    :param corpus: id of the corpus, determines referent target model
    :param models: model type (lda, nmf, ...), string or a list
    :param numTop: number of topics, string or a list
    :param thresholds: list of thresholds for cosine distance
    :return:
    '''
    # setup plot
    from matplotlib import pyplot as plt
    import numpy as np
    fig, ax = plt.subplots()
    ax.set_ylim([0, 1.0]); barw=0.5; cmap = plt.get_cmap('terrain')
    palette = ['b', 'k', 'g', 'y', 'c', 'm']
    figid = '%s_%s_T[%s]_thresh[%s]' % (corpus, models, numTop, thresholds)
    fig.suptitle(figid)
    N = len(thresholds); barx = np.arange(N)
    ax.set_xticklabels(thresholds)
    ax.set_xticks(barx+barw/2)
    ax.yaxis.grid(True)
    # fetch models
    if corpus == 'us_politics': target = resolve('gtar_themes_model')
    elif corpus == PHENO_CORPUS: target = resolve(PHENO_MODEL)
    else: raise Exception('unknown corpus: %s' % corpus)
    # plot
    if not isinstance(models, list): models = [models]
    if not isinstance(numTop, list): numTop = [numTop]
    numModels = len(models) * len(numTop); i = 0
    charts = []; chlab = []
    for model in models:
        for topics in numTop:
            modelset = baseModelSet('id', modelFolders[corpus, model, topics], numModels)
            distMetrics = [TopicCoverDist(cosineDist, th) for th in thresholds]
            means = []
            for dist in distMetrics:
                scores = [ dist(target, m) for m in modelset ]
                stats = Stats(scores)
                means.append(stats.mean)
            ch = ax.bar(barx+barw/numModels*i, means,
                        width=barw/numModels, color=cmap(1.0*i/numModels))
            chlab.append('%s[%d]'%(model, topics))
            charts.append(ch)
            i += 1
    #plt.show()
    ax.legend(charts, chlab, loc=2)
    plt.tight_layout(pad=0)
    plt.savefig(figid+'.pdf')

def coverageBasemodelsPhenotype(topics=50):
    target = resolve(PHENO_MODEL)
    print 'BASEMODELS %d TOPICS' % topics
    coverageScoringExperiment(target,
                              baseModelsSets1(topics=topics,
                                              corpus=PHENO_CORPUS),
                              coverMetrics)

def coveragePlotsV1():
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #thresholds = [0.3, 0.4, 0.5]
    with modelsContext():
        plotCoverages('us_politics', ['lda', 'nmf'], [50, 100], thresholds)
        #plotCoverages(PHENO_CORPUS, ['lda', 'nmf'], [50, 100], thresholds)
        #plotCoverages('us_politics', 'nmf', 100, thresholds)

if __name__ == '__main__':
    coveragePlotsV1()
    # with modelsContext():
    #     plotCoverages('us_politics', 'lda', 50, [0.3,0.4,0.5])