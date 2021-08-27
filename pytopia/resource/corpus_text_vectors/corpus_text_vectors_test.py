from pytopia.testing import setup

from pytopia.resource.corpus_text_vectors.CorpusTextVectors import CorpusTextVectors, \
        CorpusTextVectorsBuilder as builder
from pytopia.resource.loadSave import *
from pytopia.resource.esa_vectorizer.EsaVectorizer import EsaVectorizer

from pytopia.utils.logging_utils.setup import createLogger
from os import path

def cvectorsBuildSaveLoadCompare(builder, vectorizerCls, vectorizerOpts, dir):
    '''
    Create CorpusTextVectors, save, load, and compare original and loaded index.
    '''
    vec = vectorizerCls(**vectorizerOpts)
    cvectors = builder(vec);
    saveDir = path.join(dir, cvectors.id)
    saveResource(cvectors, saveDir)
    tfidfIndexLoad = loadResource(saveDir)
    assert cvectors == tfidfIndexLoad

def testCorpusTfidfIndexSmall(tmpdir):
    '''Test build on a small corpus. '''
    createLogger(testCorpusTfidfIndexSmall.__name__).info('STARTING TEST METHOD')
    vecOpts = {'corpus': 'us_politics_toy',
                 'dictionary': 'us_politics_toy_dict', 'text2tokens': 'RsssuckerTxt2Tokens'}
    cvectorsBuildSaveLoadCompare(builder, EsaVectorizer, vecOpts, str(tmpdir))
