'''
Build pytopia dictionaries for experiments.
'''

from doc_topic_coh.resources import pytopia_context
from pytopia.context.ContextResolver import resolve
from doc_topic_coh.settings import dataStore

from pytopia.adapt.gensim.dictionary.dict_build import \
        GensimDictBuildOptions as BuildOpts, GensimDictBuilder

def buildPlainEnglishDict(save=True):
    opts = BuildOpts(None, None, 50000)
    corpus = 'us_politics'
    txt2tokens = 'alphanum_gtar_stopword_tokenizer'
    dict = GensimDictBuilder()(corpus, txt2tokens, opts)
    dict.save(dataStore('dictionaries', dict.id))

if __name__ == '__main__':
    buildPlainEnglishDict()

