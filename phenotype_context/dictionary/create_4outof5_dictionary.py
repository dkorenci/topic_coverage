from phenotype_context.settings import corpusLabels, tfidf_matrices_folder
from pytopia.resource.loadSave import saveResource, loadResource
from pytopia.dictionary.basic_dictionary import BasicDictionary

import arff, codecs
from os import path

# todo solve this functionality within the utils package to avoid retyping
from pyutils.file_utils.location import FolderLocation as loc
thisfolder = loc(path.dirname(__file__))

# short label for '4 out of 5' dictionary
DICT_ID = 'pheno_dict1'

def readWords(fname):
    '''
    Read words contained as numeric attributes in an .arff files.
    :return: list of words
    '''
    #data = arff.load(open(fname, 'rb', ))
    data = arff.load(codecs.open(fname, 'r', 'utf-8'))
    return [ att[0] for att in data['attributes'] if att[1] == 'NUMERIC' ]

def buildDictionary():
    '''
    Create set of words that occurr in at least 4 out of 5 corpora.
    '''
    allWords = []
    for corpus in corpusLabels:
        fname = path.join(tfidf_matrices_folder, corpus+'.arff')
        words = readWords(fname)
        allWords.extend(words)
        print corpus, len(words)
    allWords = set(allWords)
    return allWords

def createDictResource(dictLabel=DICT_ID):
    '''
    Create and save pytopia dictionary.
    '''
    words = buildDictionary()
    print len(words)
    dict_ = BasicDictionary(label=dictLabel)
    dict_.addTokens(words)
    dict_.id = dictLabel
    dict_.save(thisfolder(dictLabel))

def loadDictionary():
    return loadResource(thisfolder(DICT_ID))

def test(dictLabel=DICT_ID):
    dict_ = loadResource(thisfolder(dictLabel))
    print dict_.id
    print dict_.sid
    print len(dict_)
    print ','.join(tok for tok in dict_)

def testWordsInDict(words):
    dict = loadDictionary()
    for w in words.split():
        print '%s in dict: %s' % (w, w in dict)

if __name__ == '__main__':
    #createDictResource()
    #test()
    testWordsInDict('chemoautotroph')