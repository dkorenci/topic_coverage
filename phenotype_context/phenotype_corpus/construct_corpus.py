# -*- coding: utf-8 -*-

'''
Code for construction of a corpus derived from the basic (raw) corpus
by tokenizing and stemming text and keeping only those words that
occur in the original dictionary.
'''

from os import path

from pyutils.file_utils.location import FolderLocation as loc
from phenotype_context.dictionary_context import phenotypeDictContext
from phenotype_context.phenotype_corpus.pheno_text import PhenoText
from phenotype_context.phenotype_corpus.raw_corpus import getCorpus as rawCorpus
from phenotype_context.tokenization.text2tokens_context import text2TokensContext
from pytopia.corpus.text.TextPerLineCorpus import TextPerLineCorpus, formatTextAsLine

thisfolder = loc(path.dirname(__file__))

# short label for '4 out of 5' corpus
CORPUS_ID = 'pheno_corpus1'


def constructCorpus(corpusId=CORPUS_ID):
    '''
    Create new corpus where documents text consist of those stemmed alphanumeric sequences
        derived from words of original texts that appear in pheno_dict1 dictionary
        containing tokens (stems) from the original experiment that occur in
        4 out of 5 corpora (with texts preprocessed as in the original experiment).
    Format corpus as text-per-line corpus contained in a single text-file.
    '''
    import codecs, copy
    from phenotype_context.dictionary.create_4outof5_dictionary import DICT_ID
    corpus = rawCorpus()
    txt2tok = text2TokensContext()['PhenotypeText2Tokens']
    dict = phenotypeDictContext()[DICT_ID]
    allWords = set()
    fname = thisfolder(corpusId+'.txt')
    with codecs.open(fname, 'w', 'utf-8') as cfile:
        for i, txto in enumerate(corpus):
            tokens = txt2tok(txto.text)
            tokens = [tok for tok in tokens if tok in dict]
            if tokens:
                txtcpy = copy.copy(txto)
                txtcpy.text = u' '.join(tokens)
                cfile.write(formatTextAsLine(txtcpy))
                cfile.write('\n')
            for t in tokens: allWords.add(t)
            #print ' '.join(tokens)
            #if i == 9: break
    dset = set(t for t in dict)
    if allWords != dset:
        print len(dset), len(allWords)
        print ','.join(t for t in dset.difference(allWords))

def getCorpus():
    fname = thisfolder(CORPUS_ID + '.txt')
    return TextPerLineCorpus(fname, id=CORPUS_ID, textClass=PhenoText)

def analyzeTextPreprocessing(numTexts=10, rndSeed=231):
    import random
    corpus = rawCorpus()
    txt2tok = text2TokensContext()['PhenotypeText2Tokens']
    dict = phenotypeDictContext()['pheno_dict1']
    texts = [ txto for txto in corpus ]
    random.seed(rndSeed)
    ts = random.sample(texts, numTexts)
    for txto in ts:
        tokens = txt2tok(txto.text)
        print txto.id
        print ' '.join(txto.text.split())
        print ' '.join(tokens)
        print

def testCorpus():
    from pytopia.testing.validity_checks import checkCorpus
    corpus = getCorpus()
    checkCorpus(corpus)
    #print len(corpus)
    #corpus.testLines()

if __name__ == '__main__':
    #constructCorpus()
    testCorpus()
    #analyzeTextPreprocessing(20)