import pytopia.testing.setup
from pytopia.testing.corpora import *
from pytopia.testing.utils import assertModelsEqual
from pytopia.context.GlobalContext import printGlobal
from pytopia.resource.loadSave import loadResource, saveResource

from pytopia.adapt.gensim.lda.builder import GensimLdaModelBuilder, \
    GensimLdaOptions as Opts

def builderParams():
    return [
        {'corpus':corpus_uspol_small(), 'dictionary':'us_politics_dict',
         'text2tokens':'english_word_tokenizer',
         'options':Opts(numTopics=3, alpha=50.0/3, eta=0.01, offset=1.0,
                            decay=0.5, chunksize=5, passes=10)
         }
    ]

from os import path
def testGensimBuildSaveLoad(tmpdir):
    builder = GensimLdaModelBuilder()
    for params in builderParams():
        model = builder(**params)
        assert model
        mfolder = str(tmpdir.join(model.id))
        print 'tmpmodf', mfolder
        saveResource(model, mfolder)
        modelLoad = loadResource(mfolder)
        assertModelsEqual(model, modelLoad)

if __name__ == '__main__' :
    testGensimBuildSaveLoad()

