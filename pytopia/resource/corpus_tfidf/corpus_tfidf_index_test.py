from pytopia.testing.setup import *

from pytopia.resource.corpus_tfidf.CorpusTfidfIndex import CorpusTfidfBuilder as Builder
from pytopia.resource.loadSave import *

from pytopia.utils.logging_utils.setup import createLogger
from os import path

def tfidfBuildSaveLoadCompare(builder, opts, dir):
    '''
    Create CorpusTfidfIndex, save, load, and compare original and loaded index.
    '''
    tfidfIndex = builder(**opts); saveDir = path.join(dir, tfidfIndex.id)
    saveResource(tfidfIndex, saveDir)
    tfidfIndexLoad = loadResource(saveDir)
    assert tfidfIndex == tfidfIndexLoad

def testCorpusTfidfIndexSmall(tmpdir):
    '''Test build on a small corpus. '''
    createLogger(testCorpusTfidfIndexSmall.__name__).info('STARTING TEST METHOD')
    buildOpts = {'corpus': 'us_politics_dedup_[100]_seed[1]',
                 'dictionary': 'us_politics_dict', 'text2tokens': 'english_alphanum_tokenizer'}
    builder = Builder()
    tfidfBuildSaveLoadCompare(builder, buildOpts, str(tmpdir))

def testCorpusTfidfIndexLarge(tmpdir):
    '''Test build on large corpus. '''
    createLogger(testCorpusTfidfIndexLarge.__name__).info('STARTING TEST METHOD')
    buildOpts = {'corpus': 'us_politics_dedup_[2500]_seed[3]',
                 'dictionary': 'us_politics_dict', 'text2tokens': 'english_alphanum_tokenizer'}
    builder = Builder()
    tfidfBuildSaveLoadCompare(builder, buildOpts, str(tmpdir))

