'''
Factory methods for selected coherence measures.
'''

from topic_coverage.resources import pytopia_context
#from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer
from pytopia.topic_functions.coherence.coherence_builder import CoherenceBuilder
from topic_coverage.settings import function_cache_folder
from pytopia.topic_functions.cached_function import CachedFunction
from pytopia.measure.topic_distance import cosine
#from doc_topic_coh.resources.misc_resources_context import palmettoContext

from gtar_context import gtarContext

from phenotype_context.phenotype_corpus.construct_corpus import CORPUS_ID as PHENO_CORPUS_ID
from phenotype_context.dictionary.create_4outof5_dictionary import DICT_ID as PHENO_DICT_ID

from os import path
from os.path import join

functionCacheFolder = function_cache_folder
coherenceCache = join(function_cache_folder, 'coherence')
wordcohCache = join(coherenceCache, 'wordcoh')
doccohCache = join(coherenceCache, 'doccoh')

def testDocCoh(coh):
    import pickle
    from sklearn.metrics import roc_auc_score
    from gtar_context import gtarContext
    _saveFolder = path.join(path.dirname(__file__), 'doc_coh_data')
    def resourceFname(id):
        fname = '%s.pickle' % str(id)
        return path.join(_saveFolder, fname)
    def loadById(id):
        ''' Unpickle object from the dataset folder. '''
        fname = resourceFname(id)
        if path.exists(fname):
            return pickle.load(open(fname, 'rb'))
        else:
            return None
    def topicSplit(devSize=120, rndseed=78298):
        id = 'topic_split_[devSize=%d]_[seed=%d]' % (devSize, rndseed)
        res = loadById(id)
        if res is not None: return res

    def labelMatch(tlabel, label=['theme', 'theme_noise']):
        labels = label if isinstance(label, list) else [label]
        labelset = set(labels)
        if isinstance(tlabel, basestring):
            return 1 if tlabel in labelset else 0
        elif isinstance(tlabel, dict):
            for l in labels:
                if tlabel[l] == 1: return 1
            return 0
        else:
            raise Exception('illegal topic label')
    dev, test = topicSplit()
    labels = [labelMatch(tlabel) for _, tlabel in test]
    with gtarContext():
        cohs = [coh(t) for t, _ in test]
    print roc_auc_score(labels, cohs)

def coherenceMeasure(cohParams, cacheFolder, adapt=True):
    '''
    :param cohParams: params describing the type and struct of the measure
    :param cache: path to cache folder or None
    :param adapt: weather to adapt measure to be able to receive Topic objects
    '''
    cohfunc = CoherenceBuilder(cache=None, **cohParams)()
    if adapt: cohfunc = TopicParamAdapter(cohfunc)
    return CachedFunction(cohfunc, cacheFolder, saveEvery=50, verbose=True)

def topDocCohGraph(corpus='uspol', algo='communicability'):
    if corpus == 'uspol': thresh = 0.92056
    elif corpus == 'pheno': thresh = 0.82156
    cohparams = {'distance': cosine, 'weighted': False, 'center': 'mean',
     'algorithm': algo, 'vectors': 'tf-idf',
     'threshold': 50, 'weightFilter': [0, thresh], 'type': 'graph'}
    return coherenceMeasure(cohparams, doccohCache)

def topDocCohGraph2(corpus='uspol'):
    return topDocCohGraph(corpus, 'closeness')

def topDocCohDist(corpus='uspol'):
    if corpus == 'uspol':
        corpus = 'us_politics_textperline'; text2tokens = 'whitespace_tokenizer'; dict = 'us_politics_dict'
    elif corpus == 'pheno':
        corpus = PHENO_CORPUS_ID; dict = PHENO_DICT_ID; text2tokens = 'whitespace_tokenizer'
    #TODO: pytopia "case": resource resolution @ measure construction ()
    cohparams = {'distance': cosine, 'center': 'mean', 'vectors': 'probability', 'exp': 1.0,
     'threshold': 50, 'type': 'variance',
     'corpus':corpus, 'text2tokens':text2tokens, 'dict':dict}
    return coherenceMeasure(cohparams, doccohCache)

def topWordCoh(typ, corpus='uspol', additParam=None):
    '''
    Create of of top performing word coherence measures from
    "Exploring the Space of Topic Coherence Measures"
    '''
    if typ.endswith('.local'):
        local = True
        typ = typ.split('.')[0] # extract measure type
    else: local = False
    if corpus == 'uspol': index = 'wiki_docs' if not local else 'uspol_palmetto_index'
    elif corpus == 'pheno': index = 'wiki_docs_pheno' if not local else 'pheno_palmetto_index'
    defParams = {
        'npmi': { 'type':'npmi', 'standard': False, 'index': index, 'windowSize': 10},
        'uci': { 'type':'uci', 'standard': False, 'index': index, 'windowSize': 10},
        'ca': { 'type':'c_a', 'standard': False, 'index': index, 'windowSize': 5},
        'cv': { 'type':'c_v', 'standard': False, 'index': index, 'windowSize': 110},
        'cp': { 'type':'c_p', 'standard': False, 'index': index, 'windowSize': 70},
    }
    params = defParams[typ]
    if additParam:
        for p, v in additParam.iteritems(): params[p] = v
    return coherenceMeasure(params, wordcohCache)

def phenoCorpus2txt():
    from phenotype_context import phenotypeContex
    from phenotype_context.phenotype_corpus.construct_corpus import CORPUS_ID as PHENO_CORPUS_ID
    from pytopia.corpus.tools import corpus2text
    with phenotypeContex():
        corpus2text(PHENO_CORPUS_ID, 'whitespace_tokenizer')

class TopicParamAdapter():
    '''
    Adapts Topic parameter to (modelID, topicID) pair used by most coherence functions.
    '''

    def __init__(self, topicFunc):
        self._func = topicFunc
        if hasattr(topicFunc, 'id'): self.id = topicFunc.id

    def __call__(self, topic):
        if isinstance(topic, tuple): return self._func(topic)
        else: return self._func((topic.model, topic.topicId))

def phenoDistanceDistributions():
    from doc_topic_coh.evaluations.distance_distribution import docuDistStats, tfidf
    from phenotype_context import phenotypeContex
    from phenotype_context.phenotype_corpus.construct_corpus import CORPUS_ID as PHENO_CORPUS
    from phenotype_context.dictionary.create_4outof5_dictionary import DICT_ID as PHENO_DICT
    with phenotypeContex():
        vectorizer = tfidf(PHENO_CORPUS, 'whitespace_tokenizer', PHENO_DICT)
        corpus = PHENO_CORPUS_ID
        docuDistStats(vectorizers=[vectorizer], distances = [cosine], corpus=corpus,
                      savePath='/home/damir/Dropbox/projekti/doktorat/D1 eksplorativa/phd_eksperimenti/coherence_eksperimenti/distance-distrib/')

def phenoWikiTokenize(operation='process'):
    from doc_topic_coh.resources.wiki_tokenize import processPages, tokenizedWikiToTxtFile
    from phenotype_context.tokenization.text2tokens_context import PhenotypeText2Tokens
    from pytopia.nlp.text2tokens.gtar.stopwords import RsssuckerSwRemover
    if operation == 'process':
        # create table in the database containing tokenized wiki page text
        txt2tok = PhenotypeText2Tokens(swremove=RsssuckerSwRemover())
        processPages(1, 300000, txt2tok)
    elif operation == 'dump2txt':
        tokenizedWikiToTxtFile('/datafast/enwiki/wiki_en20150602_pheno_tokenization.txt')

if __name__ == '__main__':
    #npmiUspol()
    #testDocCoh(labeledCoherences['doccoh.topgraph'])
    #testDocCoh(topDocCohDist(corpus='us_politics', text2tokens='RsssuckerTxt2Tokens', dict='us_politics_dict'))
    #phenoDistanceDistributions()
    #phenoWikiTokenize('dump2txt')
    phenoCorpus2txt()
