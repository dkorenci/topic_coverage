from topic_coverage.models.subsetter import Subsetter
from topic_coverage.resources.pytopia_context import topicCoverageContext
from topic_coverage.modelbuild.modelbuild_docker_v1 import uspolBase, phenoBase

def testSubsetter():
    from topic_coverage.resources import pytopia_context
    from pytopia.adapt.gensim.lda.builder import GensimLdaModelBuilder
    ss = Subsetter('us_politics', 'us_politics_dict', 'RsssuckerTxt2Tokens',
                    builder=GensimLdaModelBuilder, builderParams=[], buildId='testbuild', numIter=2,
                    tmpFolder='')
    print ss.id

def testSubsetterBuildLda():
    from topic_coverage.resources import pytopia_context
    from topic_coverage.modelbuild.modelbuild_iter1 import modelsContext
    from pytopia.adapt.gensim.lda.builder import GensimLdaModelBuilder, GensimLdaOptions
    bulildParams = [
    {
        'options': GensimLdaOptions(numTopics=50, alpha=1.0, eta=0.01, offset=1.0,
                         decay=0.5, chunksize=1000, passes=5, seed=3245+i)
    } for i in range(5)
    ]
    ss = Subsetter('us_politics', 'us_politics_dict', 'RsssuckerTxt2Tokens',
                    builder=GensimLdaModelBuilder(), builderParams=bulildParams,
                   buildId='testbuild', numIter=2, tmpFolder='/datafast/topic_coverage/subsetter_tmp/')
    with modelsContext():
        ss.build()
    print ss.id
    return ss
    # print type(ss), ss.__class__.__name__
    # ss.save('/datafast/topic_coverage/subsetter_tmp/test_save')
    # ssl = loadResource('/datafast/topic_coverage/subsetter_tmp/test_save')
    # print ssl.id

def generateRndseedParams(basicParams, numModels, initSeed):
    '''
    Create list of modelbuild params from basic params, by varying random seed
    '''
    from pytopia.tools.parameters import flattenParams as fp, joinParams as jp, IdList
    from copy import copy
    if isinstance(basicParams, list):
        bparams = copy(basicParams[0])
        for i in range(1, len(basicParams)):
            bparams.update(basicParams[i])
    else: bparams = copy(basicParams)
    opts = {'rndSeed': [initSeed+i for i in range(numModels)] }
    params = IdList(jp(fp(bparams), fp(opts)))
    return params

def nmfBuildParams(corpus='uspol', T=50, numModels = 5, initSeed = 5661):
    basicParams = [ uspolBase if corpus == 'uspol' else phenoBase ]
    basicParams.append({'T': T})
    params = generateRndseedParams(basicParams, numModels, initSeed)
    params.id = 'nmfSklearnParams_corpus[%s]_T[%d]_initSeed[%d]' % (corpus, T, initSeed)
    return params

def testSubsetterBuildNmf(T=50):
    from pytopia.adapt.scikit_learn.nmf.adapter import SklearnNmfBuilder
    from topic_coverage.modelbuild.modelbuild_iter1 import nmfSklearnUsPoliticsParams
    bparams = nmfBuildParams(corpus='uspol', T=T, numModels=5)
    print bparams
    for p in bparams: print p
    ss = Subsetter('us_politics_textperline', 'us_politics_dict', 'whitespace_tokenizer',
                   builder=SklearnNmfBuilder(), builderParams=bparams, paramsId='nmfTestbuildParams',
                   buildId='testbuildNmf', numIter=2, tmpFolder='/datafast/topic_coverage/subsetter_tmp/')
    ss.build()
    print ss.id
    return ss

def coverageExperiment():
    from topic_coverage.experiments.clustered_models_v0 import \
        coverageScoringExperiment, coverMetrics

    from pytopia.context.ContextResolver import resolve
    from pytopia.measure.avg_nearest_dist import AverageNearestDistance, TopicCoverDist
    from pytopia.measure.topic_distance import cosine as cosineDist

    coverMetrics = [
        AverageNearestDistance(cosineDist, pairwise=False),
        TopicCoverDist(cosineDist, 0.4)
    ]

    target = resolve('gtar_themes_model')
    models = [testSubsetterBuildNmf(100)]
    #models = [testSubsetterBuildLda()]
    coverageScoringExperiment(target, [models], coverMetrics)

if __name__ == '__main__':
    with topicCoverageContext():
        #testResampledCorpus('us_politics')
        #testSubsetterBuildLda()
        testSubsetterBuildNmf()
        #coverageExperiment()
