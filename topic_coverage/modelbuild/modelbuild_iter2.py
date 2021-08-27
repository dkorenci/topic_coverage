from topic_coverage.resources import pytopia_context

from pytopia.tools.parameters import flattenParams as fp, joinParams as jp, IdList
from pytopia.adapt.hca.HcaAdapter import HcaAdapter, HcaAdapterBuilder
from pytopia.resource.loadSave import loadResource
from pytopia.resource.builder_cache.ResourceBuilderCache import ResourceBuilderCache

from topic_coverage.settings import resource_folder
from os import path
from pyutils.file_utils.location import FolderLocation as loc
modelfolder = loc(path.join(resource_folder, 'test_models'))

uspolBase = { 'corpus':'us_politics', 'dictionary':'us_politics_dict', 'text2tokens':'RsssuckerTxt2Tokens' }
hcaBase = {
    'hcaLocation': '/data/code/hca/HCA-0.63/hca/hca',
    'tmpFolder': '/datafast/topic_coverage/modelbuild/testhca/tmp/',
    'threads': 1,
}

def addHcaBase(params): return jp(jp([uspolBase], [hcaBase]), fp(params))

pypParams = { 'type':'pyp', 'T':200, 'C':500, 'eC':100, 'rseed':[1,2,3,4,5] }
pypParams = addHcaBase(pypParams)
pypQuickTrain = { 'type':'pyp', 'T':200, 'C':100, 'eC':50, 'rseed':range(10,15) }
pypQuickTrain = addHcaBase(pypQuickTrain)
# after added sampling of all PYP hyperparams and burnin
pypV2Params = { 'type':'pyp', 'T':200, 'C':200, 'eC':50, 'rseed':range(15,20), 'burnin':30}
pypV2Params = addHcaBase(pypV2Params)

pypV2Params2 = { 'type':'pyp', 'T':200, 'C':500, 'eC':100, 'rseed':range(15,20), 'burnin':50}
pypV2Params2 = addHcaBase(pypV2Params2)

pypV2Params3 = { 'type':'pyp', 'T':500, 'C':500, 'eC':100, 'rseed':range(15,20), 'burnin':50}
pypV2Params3 = addHcaBase(pypV2Params3)

def testBuildHca():
    params = { 'corpus':'us_politics', 'dictionary':'us_politics_dict',
               'text2tokens':'RsssuckerTxt2Tokens', 'type':'pyp', 'T':50,
               'hcaLocation':'/data/code/hca/HCA-0.63/hca/hca',
                'tmpFolder':'/datafast/topic_coverage/test_hca/',
               'C':30, 'eC':30, 'burnin':10,
               }
    hca = HcaAdapter(**params)
    hca.build()
    # save/load
    svf = '/datafast/topic_coverage/test_hca/modelsave'
    hca.save(svf)
    hca2 = loadResource(svf)
    assert hca.id == hca2.id
    # basic TM interface, on loaded model
    hca = hca2
    print hca.numTopics(), hca.topicIds()
    print hca.topicVector(hca.topicIds()[3]).sum()
    for i in hca.topicIds(): print hca.topic2string(i, 15)
    print hca.topicMatrix().shape, hca.corpusTopicVectors().shape

def buildLoadTestModels(params, buildFolder):
    '''
    :param params: list of model hyperparams
    :param buildFolder: build folder label
    :return:
    '''
    from topic_coverage.settings import resource_folder
    folder = path.join(resource_folder, 'modelbuild/testhca', buildFolder)
    builder = ResourceBuilderCache(HcaAdapterBuilder, folder)
    models = []
    for p in params:
        print p
        m = builder(**p)
        models.append(m)
    return models

if __name__ == '__main__':
    #testBuildHca()
    buildLoadTestModels(pypV2Params3, 'pyp')
