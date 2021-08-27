from pytopia.context.ContextResolver import resolve

from pytopia.topic_cluster.clustered_topics_model import \
        ClusteredTopicsModelBuilder, ClusteredTopicsModel

def basicTest():
    from pytopia.topic_cluster.testing import runClusteredModelTest
    from pytopia.topic_cluster.affprop_sklearn import models4testing
    params = {
        'clusterer': models4testing()[0],
        'aggregate': 'center'
    }
    runClusteredModelTest(ClusteredTopicsModel, verbose=True, **params)

def testSaveLoadCompare(tmpdir, capsys):
    print str(tmpdir)
    from pytopia.testing.utils import createSaveLoadCompare
    from pytopia.testing import setup
    from pytopia.topic_cluster.affprop_sklearn import models4testing
    models = resolve(*['model1', 'nmf_model1'])
    params = []
    for cl in models4testing():
        params.extend([
            {'topics':models, 'topicsId':'test_models',
             'clusterer': cl, 'aggregate':'average'},
            {'topics': models, 'topicsId': 'test_models',
             'clusterer': cl, 'aggregate': 'center'}
        ])
    with capsys.disabled(): print
    for p in params:
        createSaveLoadCompare(ClusteredTopicsModelBuilder, p, str(tmpdir))
        with capsys.disabled():
            print 'createSaveLoadCompare passed:', ClusteredTopicsModelBuilder.resourceId(**p)
            #print ClusteredTopicsModelBuilder(**p)

if __name__ == '__main__':
    basicTest()