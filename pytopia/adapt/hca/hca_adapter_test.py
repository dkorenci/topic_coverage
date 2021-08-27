import pytopia.testing.setup
from pytopia.testing.corpora import *
from pytopia.testing.utils import createSaveLoadCompare
from pytopia.tools.parameters import flattenParams as fp, joinParams as jp

from pytopia.adapt.hca.HcaAdapter import HcaAdapterBuilder

import os
from os import path
from copy import copy

hcaLocation = '/data/code/hca/HCA-0.63/hca/hca' # path 2 hca executable

resourceBaseSmall = {
    'corpus': corpus_uspol_small(), 'dictionary': 'us_politics_dict',
    'text2tokens': 'english_word_tokenizer',
}
resourceBaseMed = {
    'corpus': corpus_uspol_medium(), 'dictionary': 'us_politics_dict',
    'text2tokens': 'english_word_tokenizer',
}
hcaBase = {
    'hcaLocation': '/data/code/hca/HCA-0.63/hca/hca', 'tmpFolder': None,
    'threads': 1,
}
def addBase(modelbuild, resource, base):
    '''
    Add resource and base params to modelbuild params
    '''
    if isinstance(modelbuild, dict): modelbuild = fp(modelbuild)
    return jp(jp([resource], [base]), modelbuild)

hcaModelVariants = [
    {'type':'lda', 'T':50, 'C':20, 'burnin':10, 'Cme':10, 'Bme':5},
    {'type':'lda-asym', 'T':50, 'C':20, 'burnin':10, 'Cme':10, 'Bme':5},
    {'type':'hdp', 'T':50, 'C':15, 'burnin':5, 'Cme':10, 'Bme':5},
    {'type':'pyp-doctop', 'T':50, 'C':10, 'burnin':5, 'Cme':10, 'Bme':5},
    {'type':'pyp', 'T':50, 'C':10, 'burnin':5, 'Cme':10, 'Bme':5},
]

def testHcaAdapterSmallCorpus(tmpdir):
    params = addBase(hcaModelVariants, resourceBaseSmall, hcaBase)
    runHcaAdapterSaveLoadCompare(str(tmpdir), params)

def testHcaAdapterMediumCorpus(tmpdir):
    params = addBase(hcaModelVariants, resourceBaseMed, hcaBase)
    runHcaAdapterSaveLoadCompare(str(tmpdir), params)

def runHcaAdapterSaveLoadCompare(tmpdir, params):
    hcaTmp = path.join(tmpdir, 'hcatmp'); os.mkdir(hcaTmp)
    saveDir = path.join(tmpdir, 'modelsave'); os.mkdir(saveDir)
    #for p in params: p['tmpFolder'] = hcaTmp
    createSaveLoadCompare(HcaAdapterBuilder, params, saveDir)

if __name__ == '__main__' :
    testHcaAdapterSmallCorpus()
