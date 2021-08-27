from pytopia.topic_cluster.generic_functionality import Cluster

from pytopia.context.ContextResolver import resolve

def runTopicModelsTest(Clusterer, verbose=False, **params):
    '''
    Run clustering on topics of larger testing models.
    Test is passed if the clustering does not crash.
    :param Clusterer: TopicClusterer class
    :param params: constructor params
    '''
    from pytopia.testing import setup
    #print GlobalContext.get()
    apc = Clusterer(**params)
    models = resolve('model1', 'nmf_model1')
    res = apc(models)
    validateClustering(res)
    if verbose:
        for c in res:
            print c
            for t in c: print t.id
    return res

def runClusteredModelTest(ModelClass, verbose=False, **params):
    from pytopia.testing import setup
    testModelIds = ['model1', 'nmf_model1']
    models = resolve(*testModelIds)
    params['topics'] = models
    params['topicsId'] = 'test_models'
    clustModel = ModelClass(**params)
    clustModel.build()
    if verbose:
        print clustModel.id
        print clustModel

def validateClustering(cl):
    '''
    Validate that cl is an iterable of Cluster objects
    :param cl:
    :return:
    '''
    for c in cl: assert isinstance(c, Cluster)
