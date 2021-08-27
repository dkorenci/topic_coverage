from pytopia.topic_model.ArtifTopicModel import ArtifTopicModel
from pytopia.context.ContextResolver import resolve
from phenotype_context.dictionary.create_4outof5_dictionary import DICT_ID, loadDictionary
from phenotype_context.phenotype_topics.read_topics import loadTable, loadParse
from phenotype_context.phenotype_corpus.construct_corpus import CORPUS_ID
from phenotype_context.phenotype_topics.doc_topics import createDocTopicMatrix, corpusIdIndex

from os import path
import numpy as np

from pyutils.file_utils.location import FolderLocation as loc
thisfolder = loc(path.dirname(__file__))

MODEL_ID = 'pheno_reftopics'
MODEL_DOCS_ID = 'pheno_reftopics_docs'

def normalize2prob(d, inPlace=True):
    '''
    Normalize ndarray vector or matrix rows to probability (1-sum) vectors
    '''
    from sklearn.preprocessing import normalize
    if np.ndim(d) == 1:
        r = normalize(d.reshape(1, -1), norm='l1', copy=not inPlace, axis=1)[0]
    elif np.ndim(d) == 2:
        r = normalize(d, norm='l1', copy=not inPlace, axis=1)
    else: raise Exception('normalization expects a matrix or a vector')
    if not inPlace: return r

def normalizeMatrix(m):
    '''
    Normalize topic matrix with the phenotype topics.
    '''
    normalize2prob(m)

def buildSaveModel(modelId):
    phenoTopics = loadTable()
    d = loadDictionary()
    N = len(phenoTopics)
    topicWordMatrix = np.zeros((N, d.maxIndex()+1), np.float32)
    missingWords = []; mwtopic = 0
    for i, ph in enumerate(phenoTopics):
        miss = 0; numw = 0
        for word, weight in ph:
            if word in d:
                topicWordMatrix[i, d.token2index(word)] = weight
                numw += 1
            else:
                missingWords.append(word)
                miss += 1
        if miss:
            print 'missing %d out of %d words' % (miss, len(ph))
            mwtopic += 1
    missingWords = set(missingWords)
    print
    print '%d topics missing words' % mwtopic
    print 'missing %d words in all topics' % len(missingWords)
    print ','.join(missingWords)
    normalizeMatrix(topicWordMatrix)
    print topicWordMatrix.sum()
    model = ArtifTopicModel(topicWordMatrix, dictionary=d)
    model.id = modelId
    model.save(thisfolder(modelId))

def buildSaveModelDocTopics(modelId, dictId, corpusId, txt2tokId):
    phenoTopics = loadTable()
    d = resolve(dictId)
    N = len(phenoTopics)
    topicWordMatrix = np.zeros((N, d.maxIndex()+1), np.float32)
    missingWords = []; mwtopic = 0
    for i, ph in enumerate(phenoTopics):
        miss = 0; numw = 0
        for word, weight in ph:
            if word in d:
                topicWordMatrix[i, d.token2index(word)] = weight
                numw += 1
            else:
                missingWords.append(word)
                miss += 1
        if miss:
            print 'missing %d out of %d words' % (miss, len(ph))
            mwtopic += 1
    missingWords = set(missingWords)
    print
    print '%d topics missing words' % mwtopic
    print 'missing %d words in all topics' % len(missingWords)
    print ','.join(missingWords)
    normalizeMatrix(topicWordMatrix)
    print topicWordMatrix.sum()
    # construct doc-topic matrix
    phenoParse = loadParse()
    assert len(phenoParse) == len(phenoTopics)
    topicDocMatrix = createDocTopicMatrix(phenoParse, corpusId)
    model = ArtifTopicModel(topicWordMatrix, dictionary=d, corpus=CORPUS_ID,
                            docTopicMatrix=topicDocMatrix, text2tokens=txt2tokId)
    model.id = modelId
    model.save(thisfolder(modelId))

def loadModel(mid=MODEL_ID):
    from pytopia.resource.loadSave import loadResource
    return loadResource(thisfolder(mid))

def test():
    from phenotype_context.dictionary_context import phenotypeDictContext
    from pytopia.context.GlobalContext import GlobalContext
    GlobalContext.set(phenotypeDictContext())
    print loadModel()

def modelBuildContext():
    from pytopia.context.Context import Context
    from phenotype_context.dictionary_context import phenotypeDictContext
    from phenotype_context.corpus_context import phenotypeCorpusContext
    from phenotype_context.tokenization.text2tokens_context import text2TokensContext
    ctx = Context('phenotype_modelbuild_context')
    ctx.merge(builderContext())
    ctx.merge(phenotypeDictContext())
    ctx.merge(phenotypeCorpusContext())
    ctx.merge(text2TokensContext())
    return ctx

def builderContext():
    from pytopia.resource.builders_context import basicBuildersContext
    ctx = basicBuildersContext(thisfolder('resource_builders'))
    return ctx

def validateDoctopicsModel():
    '''
    :return:
    '''
    with modelBuildContext():
        model = loadModel(MODEL_DOCS_ID)
        oldModel = loadModel(MODEL_ID)
        assert np.sum(model.topicMatrix() - oldModel.topicMatrix()) == 0
        for tid in model.topicIds():
            print 'WORDS:', model.topic2string(tid, 15)
            titles = model.topTopicDocs(tid, 15)
            texts = model.topTopicDocs(tid, 15, titles=False)
            for i, t in enumerate(titles):
                print texts[i].id, t
            print

def buildSaveDoctopicsModel():
    with modelBuildContext():
        buildSaveModelDocTopics(MODEL_DOCS_ID, DICT_ID, CORPUS_ID, 'whitespace_tokenizer')

if __name__ == '__main__':
    #buildSaveModel(MODEL_ID)
    #test()
    #buildSaveDoctopicsModel()
    validateDoctopicsModel()
    #with modelBuildContext(): corpusIdIndex(CORPUS_ID)