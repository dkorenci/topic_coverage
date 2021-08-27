'''
code handling the data manipulation for sample1
'''
from experiments.labeling.tools import *
from experiments.labeling.analyze import *
from resources.resource_builder import *
import experiments.labeling.labelsets as labelsets

SEED = 9126641

def createSample():
    sample = getCorpusIdSample(SEED, 5000)
    saveCorpusIdSample(sample, 'sample1_ids')

def printTextsTest():
    printTextRange(0, 20, 'test')

def getIndexedTexts(idSample, start, end):
    'get pairs (index, text) for text ids from the sample, in range [start, end> '
    sample = loadCorpusIdSample(idSample)
    index2id = { i:sample[i] for i in range(start, end) }
    corpus = CorpusFactory.getCorpus('us_politics')
    id2text = { txtid:txto for txtid, txto in corpus.getTexts(sample[start:end]) }
    return [ (i, id2text[index2id[i]]) for i in range(start, end) ]

def printTextRange(start, end, label, filename = None):
    indTexts = getIndexedTexts('sample1_ids', start, end)
    if filename is None:
        if label != '' : label = '_'+label
        filename = 'labeled_texts_%d-%d%s.txt'%(start,end,label)
    labels = labelsets.labels_rights(); labels.append('other rights')
    tfidfIndex = loadTfidfIndex('us_politics')
    printIndexedTextsForLabeling(indTexts, labeling_folder+filename, labels, (start, end), tfidfIndex)

def labeledFileStats(fname):
    parse = parseLabeledTexts(labeling_folder+fname)
    countLabels(parse)

def labeledFilesCompare():
    parse1 = parseLabeledTexts(labeling_folder+'labeled/labeled_texts_0-200_IAE_damir.txt')
    parse2 = parseLabeledTexts(labeling_folder+'labeled/labeled_texts_0-200_IAE_ristov.txt')
    printDifferences(parse1, parse2, labeling_folder+'diff.txt')
    #countLabels(parse)

def generateTextsForLabeling():
    ranges = [(s,s+200) for s in range(2000,3000,200)]
    for start, end in ranges:
        printTextRange(start, end, 'damir')

def generateThemeCoverageSample(start,end):
    'generate text sample for labeling document theme coverage'
    fileName = 'themecov_sample_%d-%d.txt'%(start,end)
    indTexts = getIndexedTexts('sample1_ids', start, end)
    labels = labelsets.labels_themecov();
    printIndexedTextsForLabeling(indTexts, labeling_folder+fileName, labels, (start, end))

