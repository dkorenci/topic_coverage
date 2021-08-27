from pytopia.measure.avg_nearest_dist import AverageNearestDistance
from pytopia.measure.topic_distance import cosine as cosineDist

from pytopia.context.ContextResolver import resolve
from pytopia.resource.loadSave import loadResource

from pytopia.topic_cluster.clustered_topics_model import ClusteredTopicsModelBuilder
from pytopia.topic_cluster.affprop_sklearn import AffpropSklearnTopicCluster
from pytopia.topic_cluster.hac_scipy import HacScipyTopicCluster
from pytopia.topic_cluster.kmedoid import KmedoidTopicCluster

from file_utils.location import FolderLocation as loc
from topic_coverage.settings import resource_folder

from topic_coverage.resources import pytopia_context

def addTestingModelsToContext():
    from pytopia.context.GlobalContext import GlobalContext
    from pytopia.context.Context import Context
    ctx = Context('testing_models_context')
    mfolders = []
    mfolders.extend(loc('/datafast/topic_coverage/test_models/'
                        'gensimLdaUsPoliticsParams_T[50]_initSeed[3245]_alpha[1.000]/').subfolders())
    mfolders.extend(loc('/datafast/topic_coverage/test_models/'
                        'nmfSkelarnUsPoliticsParams_T[50]_initSeed[5661]/').subfolders())
    for mf in mfolders:
        ctx.add(loadResource(mf))
    GlobalContext.get().merge(ctx)

def andAnalysis(m1, m2, andist, printAlltopics=True):
    ''' calculate AND and print results '''
    print 'model1: %d topics, %s' % (m1.numTopics(), m1.id)
    print 'model2: %d topics, %s' % (m2.numTopics(), m2.id)
    print 'AND: %.4f' % andist(m1, m2)
    if printAlltopics:
        topics = [ti for ti in m1.topicIds()]
        topics.sort(key=lambda ti: andist.nearestDist_[ti], reverse=True)
        for i, t1 in enumerate(topics):
            print '  ndist  [%2d]  %.4f' % (i, andist.nearestDist_[t1])
            print '  topic  [%2s]: %s' % (str(t1), m1.topic2string(t1, 20))
            nt = andist.nearestTopic_[t1]
            print '  ntopic [%2s]: %s' % (str(nt), m2.topic2string(nt, 20))

def testAndExperiment():
    addTestingModelsToContext()
    avgnd = AverageNearestDistance(cosineDist, pairwise=True)
    target = resolve('gtar_themes_model')
    #mfolder = loc('/datafast/topic_coverage/test_models/gensimLdaUsPoliticsParams_T[50]_initSeed[3245]_alpha[1.000]/')
    mfolder = loc('/datafast/topic_coverage/test_models/nmfSkelarnUsPoliticsParams_T[50]_initSeed[5661]/')
    models = mfolder.subfolders()[:15]
    #clusterer = AffpropSklearnTopicCluster(similarity=cosineSim, preference=0.95)
    #clusterer = HacScipyTopicCluster(distance=cosineDist, linkage='average', numClusters=100)
    clusterer = KmedoidTopicCluster(distance=cosineDist, numClusters=100)
    # source = ClusteredTopicsModelBuilder(
    #             topics=models, topicsId='test_gtar_lda', clusterer=clusterer, aggregate='average')
    source = loadResource(models[1])
    #print source
    #print target
    andAnalysis(target, source, avgnd)

if __name__ == '__main__':
    testAndExperiment()

