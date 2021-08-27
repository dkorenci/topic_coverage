from topic_coverage.resources import pytopia_context

from pytopia.adapt.gensim.lda.builder import GensimLdaModelBuilder, GensimLdaOptions
from pytopia.adapt.scikit_learn.nmf.adapter import SklearnNmfBuilder, SklearnNmfTmAdapter
from pytopia.tools.parameters import flattenParams as fp, joinParams as jp, IdList

from topic_coverage.settings import resource_folder

from os import path

from pyutils.file_utils.location import FolderLocation as loc
modelfolder = loc(path.join(resource_folder, 'test_models'))

def gensimLdaUsPoliticsOpts(numModels = 10, initSeed = 3245, T = 50, alpha=1.0):
    '''
    Return list of GensimLdaOptions for building GensimLdaModel instances
    '''
    return [
        GensimLdaOptions(numTopics=T, alpha=alpha, eta=0.01, offset=1.0,
                         decay=0.5, chunksize=1000, passes=5, seed=initSeed+i)
        for i in range(numModels)
    ]

def gensimLdaPhenotypeOpts(numModels = 10, initSeed = 3245, T = 50):
    '''
    Return list of GensimLdaOptions for building GensimLdaModel instances
        on the phenotype corpus.
    '''
    return [
        GensimLdaOptions(numTopics=T, alpha=50.0/T, eta=0.01, offset=1.0,
                         decay=0.5, chunksize=500, passes=3, seed=initSeed+i)
        for i in range(numModels)
    ]


def gensimLdaUsPoliticsParams(numModels = 10, T = 50, initSeed = 3245, alpha=1.0):
    '''
    GensimLdaModelBuilder params for building models on us_politics corpus.
    '''
    basicParams = { 'corpus':'us_politics', 'dictionary':'us_politics_dict',
                    'text2tokens':'RsssuckerTxt2Tokens' }
    opts = { 'options': gensimLdaUsPoliticsOpts(numModels, initSeed, T, alpha) }
    params = IdList(jp(fp(basicParams), fp(opts)))
    params.id = 'gensimLdaUsPoliticsParams_T[%d]_initSeed[%d]_alpha[%.3f]' % \
                    (T, initSeed, alpha)
    return params

from phenotype_context.phenotype_corpus.construct_corpus import CORPUS_ID as PHENO_CORPUS_ID
from phenotype_context.dictionary.create_4outof5_dictionary import DICT_ID
basicPhenotypeParams = { 'corpus':PHENO_CORPUS_ID,
                'dictionary':DICT_ID,
                'text2tokens':'whitespace_tokenizer' }
def gensimLdaPhenotypeParams(numModels = 10, T = 50, initSeed = 3245):
    '''
    GensimLdaModelBuilder params for building models on phenotype corpus.
    '''
    basicParams = basicPhenotypeParams
    opts = { 'options': gensimLdaPhenotypeOpts(numModels, initSeed, T) }
    params = IdList(jp(fp(basicParams), fp(opts)))
    params.id = 'gensimLdaPhenotypeParams_T[%d]_initSeed[%d]' % (T, initSeed)
    return params

def nmfSklearnUsPoliticsParams(numModels = 10, T = 50, initSeed = 5661):
    '''
    SklearnNmfBuilder params for building models on us_politics corpus.
    '''
    basicParams = { 'corpus':'us_politics', 'dictionary':'us_politics_dict',
                    'text2tokens':'RsssuckerTxt2Tokens' }
    opts = { 'T': T, 'preproc': 'tf-idf',
             'rndSeed': [initSeed+i for i in range(numModels)] }
    params = IdList(jp(fp(basicParams), fp(opts)))
    params.id = 'nmfSkelarnUsPoliticsParams_T[%d]_initSeed[%d]' % (T, initSeed)
    return params

def nmfSklearnPhenotypeParams(numModels = 10, T = 50, initSeed = 5261):
    '''
    SklearnNmfBuilder params for building models on us_politics corpus.
    '''
    basicParams = basicPhenotypeParams
    opts = { 'T': T, 'preproc': 'tf-idf',
             'rndSeed': [initSeed+i for i in range(numModels)] }
    params = IdList(jp(fp(basicParams), fp(opts)))
    params.id = 'nmfSklearnPhenotypeParams_T[%d]_initSeed[%d]' % (T, initSeed)
    return params

def buildModels(builder, params):
    print 'BUILDING PARAMS %s' % params.id
    for p in params:
        rid = builder.resourceId(**p)
        mfolder = path.join(modelfolder(params.id), rid)
        if not path.exists(mfolder):
            m = builder(**p)
            m.save(modelfolder(params.id, m.id))
            assert m.id == rid
            print 'BUILT: %s' % m.id
        else: print 'EXISTS: %s' % rid


def loadModelsFromFolder(folder):
    from pytopia.resource.loadSave import loadResource
    folder = loc(folder)
    return [ loadResource(f) for f in folder.subfolders() ]

def printModels(models):
    for m in models:
        print m

uspolModelFolders = [
    '/datafast/topic_coverage/test_models/gensimLdaUsPoliticsParams_T[50]_initSeed[3245]_alpha[1.000]/',
    '/datafast/topic_coverage/test_models/nmfSkelarnUsPoliticsParams_T[50]_initSeed[5661]/',
    '/datafast/topic_coverage/test_models/gensimLdaUsPoliticsParams_T[100]_initSeed[998]_alpha[0.500]/',
    '/datafast/topic_coverage/test_models/nmfSkelarnUsPoliticsParams_T[100]_initSeed[5661]/',
]
phenoModelFolders = [
    '/datafast/topic_coverage/test_models/gensimLdaPhenotypeParams_T[50]_initSeed[3245]/',
    '/datafast/topic_coverage/test_models/nmfSklearnPhenotypeParams_T[50]_initSeed[5261]/',
    '/datafast/topic_coverage/test_models/gensimLdaPhenotypeParams_T[100]_initSeed[3245]/',
    '/datafast/topic_coverage/test_models/nmfSklearnPhenotypeParams_T[100]_initSeed[5261]/',
]
allModelFolders = uspolModelFolders + phenoModelFolders

def loadModels(modelFolders):
    ''' Load and return all built models '''
    from pytopia.resource.loadSave import loadResource
    if not isinstance(modelFolders, list): modelFolders = [modelFolders]
    mfolders = []
    for mf in modelFolders:
        mfolders.extend(loc(mf).subfolders())
    return [loadResource(mf) for mf in mfolders]

_mctx = None
def modelsContext():
    global _mctx
    if _mctx is None:
        from pytopia.context.Context import Context
        ctx = Context('testing_models_context')
        for m in loadModels(allModelFolders): ctx.add(m)
        _mctx = ctx
    return _mctx

def addModelsToGlobalContext():
    ''' Add built models to global pytopia contex. '''
    from pytopia.context.GlobalContext import GlobalContext
    GlobalContext.get().merge(modelsContext())

def testLdaBuildUsPolitics():
    buildModels(GensimLdaModelBuilder(),
                gensimLdaUsPoliticsParams(15, T=100, initSeed=998, alpha=0.5))

def testLdaBuildPhenotype():
    buildModels(GensimLdaModelBuilder(), gensimLdaPhenotypeParams(15, T=100))

def testNmfBuildUsPolitics():
    buildModels(SklearnNmfBuilder(), nmfSklearnUsPoliticsParams(15, T=100))

def testNmfBuildPhenotype():
    buildModels(SklearnNmfBuilder(), nmfSklearnPhenotypeParams(15, T=100))

# def testNmfBuildUsPolitics():
#     buildModels(SklearnNmfBuilder(), nmfSklearnUsPoliticsParams(5))

if __name__ == '__main__':
    #testLdaBuildUsPolitics()
    #testLdaBuildPhenotype()
    testNmfBuildUsPolitics()
    #testNmfBuildPhenotype()
    #printModels(loadModels('/datafast/topic_coverage/test_models/gensimLdaPhenotypeParams_T[50]_initSeed[3245]/'))
    #printModels(loadModels('/datafast/topic_coverage/test_models/gensimLdaUsPoliticsParams_T[50]_initSeed[3245]_alpha[1.000]/'))
    #printModels(loadModels('/datafast/topic_coverage/test_models/nmfSkelarnUsPoliticsParams_T[50]_initSeed[5661]/'))
