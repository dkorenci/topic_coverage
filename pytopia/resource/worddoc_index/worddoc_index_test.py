from pytopia.testing.setup import *
from pytopia.context.ContextResolver import resolve
from pytopia.resource.worddoc_index.worddoc_index import WordDocIndexBuilder as Builder
from pytopia.resource.loadSave import *

from pytopia.utils.logging_utils.setup import createLogger
from os import path

def wdiBuildSaveLoadCompare(builder, opts, dir):
    '''
    Create CorpusTfidfIndex, save, load, and compare original and loaded index.
    '''
    wdi = builder(**opts); saveDir = path.join(dir, wdi.id)
    saveResource(wdi, saveDir)
    wdiLoad = loadResource(saveDir)
    assert wdi == wdiLoad

def testWordDocIndexSmall(tmpdir):
    '''Test build on a small corpus. '''
    createLogger(testWordDocIndexSmall.__name__).info('STARTING TEST METHOD')
    buildOpts = {'corpus': 'us_politics_dedup_[100]_seed[1]',
                 'dictionary': 'us_politics_dict', 'text2tokens': 'english_alphanum_tokenizer'}
    builder = Builder()
    wdiBuildSaveLoadCompare(builder, buildOpts, str(tmpdir))

def testWordDocIndexLarge(tmpdir):
    '''Test build on large corpus. '''
    createLogger(testWordDocIndexLarge.__name__).info('STARTING TEST METHOD')
    buildOpts = {'corpus': 'us_politics_dedup_[2500]_seed[3]',
                 'dictionary': 'us_politics_dict', 'text2tokens': 'english_alphanum_tokenizer'}
    builder = Builder()
    wdiBuildSaveLoadCompare(builder, buildOpts, str(tmpdir))

def compareWordDocIndexDetails(corpusTxt, wordQueries, docQueries):
    '''
    Create index for a toy corpus and compare results to expected values.
    :return:
    '''
    # create pytopia corpus and add it to global context
    from pytopia.corpus.text.TextCorpus import TextCorpus
    toyCorpus = TextCorpus(corpusTxt)
    toyCorpus.id = 'toy_corpus'
    from pytopia.context.Context import Context
    # build dictionary and add it to global context
    txt2Tokens = 'english_alphanum_tokenizer'
    from pytopia.adapt.gensim.dictionary.dict_build import GensimDictBuilder, \
        GensimDictBuildOptions
    opts = GensimDictBuildOptions(None, None, None)
    toyDict = GensimDictBuilder().buildDictionary(toyCorpus, txt2Tokens, opts)
    toyDict.id = 'toy_dict'
    with Context('', toyCorpus, toyDict):
        # build word doc index
        wdi = Builder().__call__(toyCorpus, txt2Tokens, toyDict)
        for word, result in wordQueries:
            assert sorted(wdi.wordDocs(word)) == sorted(result)
        for doc, result in docQueries:
            # convert word strings to (dictionary) indexes
            wiRes = [ (toyDict.token2index(word), count) for word, count in result ]
            assert sorted(wdi.docWords(doc)) == sorted(wiRes)

def testDocWordIndexDetails1():
    corpusTxt = '''
    id = 0, text = a b c
    id = 1, text = d b e b
    id = 2, text = a b d e f g h
    id = 3, text = b f g f f f
    id = 4, text = b
    '''
    wordQueries = [
        ('b', [('0', 1), ('1', 2), ('2', 1), ('3', 1), ('4', 1)]),
        ('f', [('2', 1), ('3', 4)]),
        ('c', [('0', 1)]),
    ]
    docQueries = [
        ('3', [('b', 1), ('f', 4), ('g', 1)]),
        ('4', [('b', 1)]),
        ('1', [('d', 1), ('b', 2), ('e', 1)])
    ]
    compareWordDocIndexDetails(corpusTxt, wordQueries, docQueries)