from pytopia.testing.setup import *
from pytopia.adapt.gensim.lda.builder import GensimLdaModelBuilder, \
        GensimLdaOptions as Opts
from pytopia.adapt.scikit_learn.nmf.adapter import SklearnNmfBuilder
from pytopia.resource.loadSave import saveResource

from os import path

modelDir = path.join(path.dirname(__file__), 'models')

def buildTestModels(filter=None):
    buildParams = [
        ( 'model1', GensimLdaModelBuilder(),
        {'corpus': 'us_politics_dedup_[2500]_seed[3]',
         'dictionary':'us_politics_dict', 'text2tokens':'english_word_tokenizer',
         'options':Opts(numTopics=30, alpha=50.0/30, eta=0.01, offset=1.0,
                            decay=0.5, chunksize=5, passes=10)
         }
        ),
        ('nmf_model1', SklearnNmfBuilder(),
         {'corpus': 'us_politics_dedup_[2500]_seed[3]',
          'dictionary': 'us_politics_dict', 'text2tokens': 'english_word_tokenizer',
          'T': 30
          }
         )
    ]

    for mid, builder, p in buildParams:
        if not filter or mid in filter:
            model = builder(**p)
            model.id = mid
            saveResource(model, path.join(modelDir, str(model.id)))
            print model
            print model.id
            print model.sid

if __name__ == '__main__':
    #buildTestModels()
    buildTestModels(['nmf_model1'])