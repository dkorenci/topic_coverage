from doc_topic_coh.resources import pytopia_context

from pytopia.adapt.gensim.lda.builder import GensimLdaModelBuilder, \
    GensimLdaOptions as Opts
from pytopia.resource.loadSave import saveResource

from doc_topic_coh.evaluations.tools import topicMeasureAuc, \
    flattenParams as fp, joinParams as jp

from doc_topic_coh.settings import dataStore

def buildOptions(T, rndSeed):
    return Opts(numTopics=T, alpha=50.0 / T, eta=0.01, offset=1.0, decay=0.5,
                chunksize=1000, passes=10, seed=rndSeed)

def buildModels(modelsFolder, initSeed):
    basicParams = [
        { 'corpus': 'us_politics', 'text2tokens': 'RsssuckerTxt2Tokens',
          'dictionary' : 'us_politics_dict' }
    ]
    numTopics = [50, 100, 100]
    buildParams = [ { 'options': buildOptions(T, initSeed+i) } for i, T in enumerate(numTopics) ]
    params = jp(basicParams, buildParams)
    builder = GensimLdaModelBuilder()
    for p in params:
        print p['options'].__dict__
        model = builder(**p)
        print model.id
        mfolder = dataStore.subfolder(modelsFolder, model.id)
        saveResource(model, mfolder)

if __name__ == '__main__':
    buildModels('models1', 727841)