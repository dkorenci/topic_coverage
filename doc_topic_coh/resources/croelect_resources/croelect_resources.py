from doc_topic_coh.settings import dataStore
from pytopia.context.Context import Context
from pytopia.resource.loadSave import loadResource, saveResource
from pytopia.adapt.gensim.lda.GensimLdaModel import GensimLdaModel

from os import path, listdir

corpusId = 'iter0_cronews_final'
dictId = 'croelect_dict_iter0'
text2tokensId = 'CroelectTxt2Tokens'

def createWrappedCroelectModel(mid, folder):
    m = GensimLdaModel(None, id=mid)
    m.load(folder)
    m.corpus = corpusId
    m.dictionary = dictId
    m.text2tokens = text2tokensId
    return m

def croelectOriginalModels():
    folder = dataStore.subfolder('croelect', 'annotated_models')
    models = []
    for f in ['model1', 'model2', 'model3']:
        m = loadResource(path.join(folder,f), objectFile='TopicModel_object')
        m.id = 'croelect.%s' % f
        models.append(m)
        print m.__dict__
    return models

def createWrappedCroelectDictionary():
    ''' Wrap gensim dictionary from orig. croelect models as pytopia resource. '''
    from pytopia.adapt.gensim.dictionary.GensimDictAdapter import GensimDictAdapter
    m = croelectOriginalModels()[0]
    d = GensimDictAdapter(m.dictionary, id=dictId,
                          corpusId=corpusId, txt2tokId=text2tokensId)
    folder = dataStore.subfolder('croelect', 'annotated_models')
    saveResource(d, folder)

def croelectDictionary():
    folder = dataStore.subfolder('croelect', 'annotated_models', 'dict')
    d = loadResource(folder)
    #print d.id
    #print d.__dict__
    return d

def croelectModelsContext():
    ctx = Context('croelect_models')
    # folder = dataStore.subfolder('croelect', 'annotated_models')
    # for f in ['model1', 'model2', 'model3']:
    #     m = createWrappedCroelectModel('croelect_%s' % f, path.join(folder,f))
    #     ctx.add(m)
    folder = dataStore.subfolder('croelect', 'labels_improve')
    for f in ['model1', 'model2', 'model3', 'model4']:
        m = createWrappedCroelectModel('croelect_%s' % f, path.join(folder,f))
        ctx.add(m)
    return ctx

def croelectMiscResourceContext():
    ctx = Context('croelect_resources')
    ctx['croelect_palmetto_index'] = '/datafast/palmetto_indexes/croelect/windowed'
    ctx['crowiki_palmetto_index'] = '/datafast/hrwiki/lucene/'
    from croelect.preprocessing.stopwords import CroelectSwRemover
    from pytopia.nlp.text2tokens.gtar.text2tokens import alphanumStopwordsTokenizer
    crotok = alphanumStopwordsTokenizer(CroelectSwRemover())
    crotok.id = 'croelect_alphanum_stopword_tokenizer'
    ctx.add(crotok)
    return ctx

def croelectText2Tokens():
    from croelect.preprocessing.text2tokens import CroelectTxt2Tokens
    txt2tok = CroelectTxt2Tokens()
    txt2tok.id = text2tokensId
    return txt2tok

if __name__ == '__main__':
    #croelectModelsContext()
    #createWrappedCroelectDictionary()
    #croelectDictionary()
    #croelectOriginalModels()
    croelectText2Tokens()
