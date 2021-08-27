'''
scikit-learn datasets (numpy and scipy vectors) creation, saving and loading
'''
import os

import numpy as np

from experiments.labeling.tools import parseLabeledTexts, getLabeledTextObjectsFromParse
from resources.resource_builder import *
from pymedialab_settings.settings import object_store


def datasetsFolder(): return object_store+'classification/'
def datasetFile(datasetName):
    fname = datasetsFolder()+datasetName+'.npy'
    return fname

def loadDataset(name):
     return np.load(datasetFile(name))

def saveDataset(name, dataset):
    if not os.path.exists(datasetsFolder()): os.makedirs(datasetsFolder())
    np.save(datasetFile(name), dataset)

def constructTextClasses(fname, labels, multiLabel = False):
    'create numpy array with class labels, from labeled text file'
    # load data and resources
    parse = parseLabeledTexts(labeling_folder+fname)
    # construct indexes
    D = len(parse)
    L = len(labels)
    nolabelC = L
    l2c = { l:i for i, l in enumerate(labels) }
    # construct array
    if multiLabel: cl = np.zeros((D, L))
    else: cl = np.zeros(D)
    r = 0
    for ti, tid, lab in parse :
        if not multiLabel:
            nolabel = True
            for l in labels:
                if lab[l] == 1 :
                    cl[r] = l2c[l]
                    nolabel = False
                    break
            if nolabel :
                cl[r] = nolabelC
        else:
            for l in labels:
                if lab[l] == 1 :
                    cl[r, l2c[l]] = 1
        r += 1
    return cl


def constructTextFeatures(fname, corpus_id, txt2tok =  RsssuckerTxt2Tokens()):
    'create numpy array with tf-idf weights from texts'
    # load data and resources
    dict = loadDictionary(corpus_id)
    tfidf = loadTfidfIndex(corpus_id)
    corpus = CorpusFactory.getCorpus(corpus_id)
    parse = parseLabeledTexts(labeling_folder+fname)
    # construct indexes
    ids = []; id2lab = {}; tid2row = {}; r = 0
    for ti, tid, lab in parse :
        id2lab[tid] = lab
        ids.append(tid)
        tid2row[tid] = r; r += 1
    N = len(dict)
    D = len(ids)
    m = np.zeros((D, N))
    # construct numpy array
    for tid, txto in corpus.getTexts(ids):
        r = tid2row[tid]
        for wi, freq in dict.doc2bow(txt2tok(txto.text)):
            m[r, wi] = (1+np.log2(freq))*tfidf.tfidf.idfs[wi]
        #m[r] /= norm(m[r])
    return m

def getTextsAndLabelsFromFile(fname, label):
    labTexts = getLabeledTextObjectsFromParse(parseLabeledTexts(labeling_folder+fname))
    return getTextsAndLabelsFromFile(labTexts, label)

def getTextsAndLabels(labTexts, label):
    '''
    :param labTexts: list of (Text, labeling) pairs
    :param label: string label
    :return: list of Texts, array of binary class labels - 0 for no label, 1 for label
    '''
    texts = []
    classLabels = np.zeros(len(labTexts))
    i = 0
    for txto, labels in labTexts:
        if label not in labels:
            raise Exception('label %s is not among text labels for text %d' % (label, txto.id))
        texts.append(txto)
        classLabels[i] = labels[label]
        i += 1
    return texts, classLabels