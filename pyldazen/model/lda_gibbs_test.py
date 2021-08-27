import pytopia.testing.setup
from pytopia.testing.corpora import *
from pytopia.testing.utils import assertModelsEqual
from pytopia.context.GlobalContext import printGlobal
from pytopia.resource.loadSave import loadResource, saveResource
from pytopia.context.ContextResolver import resolve
from pytopia.context.GlobalContext import GlobalContext
from pytopia.resource.corpus_topics.CorpusTopicIndex import CorpusTopicIndexBuilder

from pyldazen.model.LdaGibbsTopicModel import LdaGibbsTopicModelBuilder


# def __init__(self, corpus, dictionary, text2tokens,
#              numTopics, alpha, beta, gibbsIter=1000,
#              fixedTopics=None, idLabel=None, rndSeed=88911):
#
def builderParams():
    return [
        # small corpus, default nmf building options
        {
         'corpus':corpus_uspol_small(), 'dictionary':'us_politics_dict',
         'text2tokens':'english_word_tokenizer', 'numTopics': 5, 'gibbsIter':100
         },
        # medium corpus, default nmf building options
        {
            'corpus': corpus_uspol_medium(), 'dictionary': 'us_politics_dict',
            'text2tokens': 'english_word_tokenizer', 'numTopics': 30, 'gibbsIter':30
        }
    ]

def testBuildSaveLoad(tmpdir):
    builder = LdaGibbsTopicModelBuilder
    for params in builderParams():
        model = builder(**params)
        print model.id
        assert model
        # todo generic functionality for corpus-topic-index tests with new models
        GlobalContext.get().add(model)
        cti = CorpusTopicIndexBuilder()(params['corpus'], model)
        assert cti is not None
        print cti.id
        mfolder = str(tmpdir.join(model.id))
        print 'tmpmodf', mfolder
        saveResource(model, mfolder)
        modelLoad = loadResource(mfolder)
        assertModelsEqual(model, modelLoad)

#todo: tests with resources depending on the model (topic index, aggregation, ...)
#todo: make this test generic and applicable to other models

if __name__ == '__main__' :
    testBuildSaveLoad()#'/datafast/topic_coverage/tmp/')

