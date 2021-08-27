import pytopia.testing.setup
from pytopia.testing.corpora import *
from pytopia.testing.utils import createSaveLoadCompare
from pytopia.tools.parameters import flattenParams as fp, joinParams as jp

from pytopia.adapt.artm.artm_adapter import ArtmAdapterBuilder

import tempfile, shutil

resourceBaseSmall = {
    'corpus': corpus_uspol_small(), 'dictionary': 'us_politics_dict',
    'text2tokens': 'english_word_tokenizer',
}
resourceBaseMed = {
    'corpus': corpus_uspol_medium(), 'dictionary': 'us_politics_dict',
    'text2tokens': 'english_word_tokenizer',
}

def addBase(modelbuild, resource):
    '''
    Add resource and base params to modelbuild params
    '''
    if isinstance(modelbuild, dict): modelbuild = fp(modelbuild)
    return jp([resource], modelbuild)

artmModelVariants = [
    {'type': 'decorr', 'T': 30, 'iter': 15, 'tau':1e+3 },
    {'type': 'SB', 'T': 30, 'iter': 15, 'specific': 0.5, 'tauSpec': 1e+4, 'tauBck':0.5},
    {'type': 'smspde', 'T': 30, 'iter': 15, 'sparseOn':5,
     'specific':0.8, 'tauSmPhi':1.0, 'tauSmTh':1.0, 'tauSpPh':-1.0, 'tauSpTh':-1.0, 'tauDec':1e+4},
]

def testHcaAdapterSmallCorpus():
    params = addBase(artmModelVariants, resourceBaseSmall)
    artmAdapterSaveLoadCompare(params)

def testHcaAdapterMediumCorpus():
    params = addBase(artmModelVariants, resourceBaseMed)
    artmAdapterSaveLoadCompare(params)

def artmAdapterSaveLoadCompare(params):
    saveDir = tempfile.mkdtemp()
    createSaveLoadCompare(ArtmAdapterBuilder, params, saveDir)
    shutil.rmtree(saveDir, ignore_errors=True)

if __name__ == '__main__' :
    testHcaAdapterMediumCorpus()
