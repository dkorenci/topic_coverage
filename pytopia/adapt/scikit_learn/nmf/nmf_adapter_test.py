import pytopia.testing.setup
from pytopia.testing.corpora import *
from pytopia.testing.utils import assertModelsEqual
from pytopia.context.GlobalContext import printGlobal
from pytopia.resource.loadSave import loadResource, saveResource
from pytopia.context.GlobalContext import GlobalContext
from pytopia.context.Context import Context
from pytopia.resource.corpus_topics.CorpusTopicIndex import CorpusTopicIndexBuilder

from pytopia.adapt.scikit_learn.nmf.adapter import SklearnNmfTmAdapter, SklearnNmfBuilder

def builderParams():
    return [
        # small corpus, default nmf building options
        {
         'corpus':corpus_uspol_small(), 'dictionary':'us_politics_dict',
         'text2tokens':'english_word_tokenizer',
         'T': 5
         },
        # medium corpus, default nmf building options
        {
            'corpus': corpus_uspol_medium(), 'dictionary': 'us_politics_dict',
            'text2tokens': 'english_word_tokenizer',
            'T': 30
        }
    ]

from os import path
def testNmfBuildSaveLoad(tmpdir):
    builder = SklearnNmfBuilder()
    for params in builderParams():
        model = builder(**params)
        assert model
        # todo generic functionality for corpus-topic-index tests with new models
        with Context('tmpctx', model):
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
    testNmfBuildSaveLoad()

