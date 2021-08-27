from pytopia.topic_model.ArtifTopicModel import ArtifTopicModel
from pytopia.context.GlobalContext import GlobalContext
from pytopia.context.ContextResolver import resolveIds, resolve
from gtar_context.semantic_topics.parse_themes import parseAllThemes

import numpy as np
from os import path

from pyutils.file_utils.location import FolderLocation as loc
thisfolder = loc(path.dirname(__file__))

__contextSet = False
def __initContext():
    from gtar_context.builders_context import buildersContext
    from gtar_context.basic_context import basicContext
    global __contextSet
    if __contextSet: return
    ctx = basicContext()
    ctx.merge(buildersContext())
    GlobalContext.set(ctx)

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

def themeWordVector(theme, dict, txt2tok):
    '''
    Create probability BoW vector from theme words.
    '''
    # keep only those words that are in the dictionary (either preprocessed or not)
    words = []
    for w in theme.words:
        if w in dict: words.append(w)
        else:
            sw = txt2tok(w)[0]
            if sw in dict: words.append(sw)
    # create topic vector from words
    words = set(words)
    if words:
        wordVector = np.zeros(len(dict), np.float32)
        for w in words:
            wordVector[dict.token2index(w)] = 1.0
        normalize2prob(wordVector)
        assert np.abs(wordVector.sum()-1.0) < 10e-7
        return wordVector

def themeDocVector(theme, dict, corpus, txt2tok):
    '''
    Create probability BoW vector from tfidf vectors of theme documents.
    '''
    docs = set([docId for docId in theme.docs])
    ldocs = list(docs)
    if docs:
        docMatrix = np.zeros((len(docs), len(dict)), np.float32)
        tfidf = resolve('corpus_tfidf_builder')(corpus, dict, txt2tok)
        for i, txto in enumerate(corpus.getTexts(docs)):
            if txto == None: print 'MISSING TEXT', ldocs[i]
            else: docMatrix[i] = tfidf[txto.id]
        normalize2prob(docMatrix) # normalize rows, tfidfs -> probabilities
        docVector = np.sum(docMatrix, 0)
        normalize2prob(docVector)
        return docVector

#requires corpus_tfidf_builder
def theme2topic(theme, dict, corpus, txt2tok):
    from gtar_context.semantic_topics.parse_themes import ParsedSemTopics
    assert isinstance(theme, ParsedSemTopics)
    wordVec = themeWordVector(theme, dict, txt2tok)
    docVec = themeDocVector(theme, dict, corpus, txt2tok)
    if theme.dominantData is None: ww, dw = 0.5, 0.5
    elif theme.dominantData == 'words': ww, dw = 0.8, 0.2
    elif theme.dominantData == 'docs': ww, dw = 0.2, 0.8
    else: raise Exception('undefined dominant data value: [%s]'%str(theme.dominantData))
    return ww*wordVec + dw*docVec

# requires corpus_index_builder
def themes2doctopic(themes, corpus):
    ci = resolve('corpus_index_builder')(corpus)
    docTopics = np.zeros((len(ci), len(themes)), np.float32)
    numMissing = 0
    for i, theme in enumerate(themes):
        for docId in theme.docs:
            #doci = ci.id2index(long(docId))
            try: doci = ci.id2index(docId)
            except:
                print 'missing docId', docId
                numMissing += 1
                print theme
                doci = None
            if doci is not None: docTopics[doci, i] = 1.0
    print 'MISSING DOCS', numMissing
    return docTopics

def buildSaveModel(modelId, dict, corpus, txt2tok):
    themes = parseAllThemes()
    dict, corpus, txt2tok = resolve(dict, corpus, txt2tok)
    topicMatrix = np.zeros((len(themes), len(dict)), np.float32)
    for i, th in enumerate(themes):
        print th
        topicMatrix[i] = theme2topic(th, dict, corpus, txt2tok)
        print i
    docTopics = themes2doctopic(themes, corpus)
    model = ArtifTopicModel(topicMatrix, corpus, dict, txt2tok, docTopicMatrix=docTopics)
    model.id = modelId
    model.save(thisfolder(modelId))
    print model.id
    print model.sid

MODEL_ID_OLD = 'gtar_themes_model_old'
def constructThemeModel():
    buildSaveModel(MODEL_ID_OLD, 'us_politics_dict', 'us_politics', 'RsssuckerTxt2Tokens')

MODEL_ID = 'gtar_themes_model'
def constructNewThemeModel():
    buildSaveModel(MODEL_ID, 'us_politics_dict', 'us_politics_textperline', 'whitespace_tokenizer')

def gtarRefModelsContext():
    from pytopia.context.Context import Context
    from pytopia.resource.loadSave import loadResource
    ctx = Context('gtar_refmodels_context')
    ctx.add(loadResource(thisfolder(MODEL_ID)))
    ctx.add(loadResource(thisfolder(MODEL_ID_OLD)))
    return ctx

def loadModel():
    from pytopia.resource.loadSave import loadResource
    return loadResource(thisfolder(MODEL_ID))

def iterateTexts():
    corpus = resolve('us_politics')
    for txto in corpus: pass

if __name__ == '__main__':
    __initContext()
    #constructThemeModel()
    constructNewThemeModel()
    #iterateTexts()