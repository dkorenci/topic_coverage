import pytest

from pytopia.adapt.gensim.dictionary.dict_build import GensimDictBuildOptions as Opts
from pytopia.adapt.gensim.dictionary.dict_build import GensimDictBuilder
from pytopia.corpus.text.TextCorpus import TextCorpus
from pytopia.nlp.text2tokens.regexp import alphanumTokenizer

def toyCorpus():
    txt = '''
    id = 0, text = a b c d
    id = 1, text = c d e f
    id = 2, text = e f g h
    '''
    return TextCorpus(txt)

def assertDictWordSet(dict, words):
    '''
    :param dict: pytopia dictionary
    '''
    dictWs = set(w for w in dict.iterkeys())
    ws = set(w for w in words)
    assert ws == dictWs

@pytest.fixture(scope='session')
def dictTestSets():
    c = toyCorpus(); t2t = alphanumTokenizer()
    # list of pairs (dict builder params, expected word set)
    return [
        ({ 'corpus':c, 'txt2tokens':t2t,
           'opts':Opts(None,None,None) },
           'a b c d e f g h'.split()),
        ({'corpus': c, 'txt2tokens': t2t,
          'opts': Opts(2, None, None)},
         'c d e f'.split()),
        ({'corpus': c, 'txt2tokens': t2t,
          'opts': Opts(None, 1, None)},
         'a b g h'.split()),
        ({'corpus': c, 'txt2tokens': t2t,
          'opts': Opts(None, None, 4)},
         'c d e f'.split())
    ]

def testGensimDictBuild(dictTestSets):
    for opts, words in dictTestSets:
        dict = GensimDictBuilder()(**opts)
        print words
        assertDictWordSet(dict, words)



