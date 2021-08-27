'''
Analyze labeled data
'''
import codecs

import numpy as np

from corpus.factory import CorpusFactory
from experiments.labeling.tools import formatText, parseLabeledTexts
from pymedialab_settings.settings import labeling_folder
from experiments.labeling.tools import getLabeledTextObjectsFromParse


def countLabels(parse):
    agglab = {}
    for i, id, lab in parse:
        for l in lab:
            if l not in agglab : agglab[l] = 0
            agglab[l] += lab[l]
    for l in agglab:
        print l, agglab[l]

def labeledSetsSizeDiff():
    file1 = 'labeled/test_ristov1000-1200_damir2000-2800_cleaned.txt' #'labeled/test1_ristov1000-1200_damir2000-2400.txt'
    file2 = 'labeled/test_ristov1000-1200_damir2000-2800.txt'
    fileLabel = 'test'
    parse1 = parseLabeledTexts(labeling_folder+file1)
    parse2 = parseLabeledTexts(labeling_folder+file2)
    print len(parse1), len(parse2)

def calculateIAA():
    file1 = 'labeled/labeled_texts_0-200_IAA_damir_corr.txt' #'labeled/test1_ristov1000-1200_damir2000-2400.txt'
    file2 = 'labeled/labeled_texts_0-200_IAA_ristov_corr.txt'
    labels = [
        'civil rights movement',
        'gay rights' ,
        'police brutality' ,
        'chapel hill' ,
        'fraternity racism' ,
        'reproductive rights' ,
        'violence against women' ,
        'death penalty' ,
        'surveillance' ,
        'gun rights' ,
        'net neutrality' ,
        'marijuana' ,
        'vaccination'
    ]
    fileLabel = 'test'
    parse1 = parseLabeledTexts(labeling_folder+file1)
    id2lmap1 = { id:lmap for ind, id, lmap in parse1 }
    ids = [ id for ind, id, lmap in parse1 ]
    parse2 = parseLabeledTexts(labeling_folder+file2)
    id2lmap2 = { id:lmap for ind, id, lmap in parse2 }
    for i in ids:
        lmap1 = id2lmap1[i]; lmap2 = id2lmap2[i]
        lmap1['other rights'] = 0; lmap2['other rights'] = 0
        c1 = 0
        for l in labels: c1 += lmap1[l];
        if c1 == 0 : lmap1['X'] = 1
        else: lmap1['X'] = 0

        c2 = 0
        for l in labels: c2 += lmap2[l]
        if c2 == 0 : lmap2['X'] = 1
        else: lmap2['X'] = 0

    labels.append('X'); L = len(labels)
    m = np.zeros((L,L))
    l2id = { l:i for i,l in enumerate(labels) }
    for i in ids:
        lmap1 = id2lmap1[i]; lmap2 = id2lmap2[i]
        c1 = 0
        for l in labels: c1 += lmap1[l];
        c2 = 0
        for l in labels: c2 += lmap2[l]

        if c1 == c2 and c1 == 1:
            for l in lmap1:
                if lmap1[l] == 1: l1 = l
            for l in lmap2:
                if lmap2[l] == 1: l2 = l
            m[l2id[l1],l2id[l2]] += 1
        else:
            print lmap1
            print lmap2

        # c1 = 0; for l in labels: c1 += lmap1[l];
        # if c1 == 0 : lmap1['X'] = 1
        # else: lmap1['X'] = 0
        # c2 = 0; for l in labels: c2 += lmap2[l]
        # if c2 == 0 : lmap2['X'] = 1
        # else: lmap2['X'] = 0
    m[l2id['X'],l2id['civil rights movement']] += 1
    m[l2id['X'],l2id['gay rights']] += 1

    print m
    s = ''
    for i in range(L):
        for j in range(L):
            s += '%d , '%int(m[i,j])

    print s
    print L

    print len(parse1), len(parse2)

def printDifferences(parse1, parse2, file):
    id1 = id2lab(parse1)
    id2 = id2lab(parse2)
    if set(id1.keys()) != set(id2.keys()):
        s1, s2 = set(id1.keys()), set(id2.keys())
        print s1.difference(s2)
        print s2.difference(s1)
        raise Exception('key sets do not match')
    corpus = CorpusFactory.getCorpus('us_politics')
    mismatchid = [ id for id in id1 if id1[id] != id2[id] ]
    mmid2text = { id:txto for id,txto in corpus.getTexts(mismatchid) }
    f = codecs.open(file, "w", "utf-8")
    f.write('NUM MISMATCHES: %d' % len(mismatchid) + '\n')
    for i, id in enumerate(mismatchid):
        ml1, ml2 = mismatchLabels(id1[id], id2[id])
        f.write('MISMATCH: %d' % i + '\n')
        f.write('L1: ' + labelingToString(ml1) + '\n')
        f.write('L2: ' + labelingToString(ml2) + '\n')
        f.write('TITLE: %s' % mmid2text[id].title + '\n')
        f.write('TEXT ID: %d\n' % id)
        f.write(formatText(mmid2text[id].text)+'\n\n')

def mismatchLabels(l1, l2):
    if set(l1.keys()) != set(l2.keys()):
        print labelingToString(l1)
        print labelingToString(l2)
        raise Exception('labels mismatch')
    return { l:l1[l] for l in l1 if l1[l] != l2[l] }, { l:l2[l] for l in l1 if l1[l] != l2[l] }

def labelingToString(lab):
    s = ''
    for l in lab: s += '%s %d ' % (l, lab[l])
    return s

def id2lab(parse):
    return { id: lab for index, id, lab in parse }

def id2parseMap(parse):
    return { id: (index, lab) for index, id, lab in parse }

def addLabelsToCount(labels, cnt):
    'add labels for one document to global numLabels -> numDocs count'
    numl = sum(1 for l in labels if labels[l] == 1)
    if numl in cnt : cnt[numl] += 1
    else: cnt[numl] = 1

def numLabelsPerDocument(labeledFile):
    parse = parseLabeledTexts(labeling_folder+labeledFile)
    doc4numl = {  }; numDocs = 0
    for _,_,labels in parse:
        numDocs += 1
        addLabelsToCount(labels, doc4numl)
    print 'num documents %d' % numDocs
    for n in doc4numl :
        print 'num lables %2d , num documents %4d' % (n, doc4numl[n])


def countNumLabelsForMapper(labeledFile, mapper):
    parse = parseLabeledTexts(labeling_folder+labeledFile)
    labTexts = getLabeledTextObjectsFromParse(parse)
    doc4numl = {  }; numDocs = 0
    for txto, _ in labTexts:
        resLabels = mapper.map(txto)
        addLabelsToCount(resLabels, doc4numl)
    print 'num documents %d' % numDocs
    for n in doc4numl :
        print 'num lables %2d , num documents %4d' % (n, doc4numl[n])

def printLabelFrequencies(labeledFile):
    'print frequencies of labels in labeled file'
    parse = parseLabeledTexts(labeling_folder+labeledFile)
    labelCounts = {}
    for _,_,labels in parse:
        for l in labels:
            if l not in labelCounts: labelCounts[l] = 0
            labelCounts[l] += labels[l]
    numFiles = float(len(parse))
    for l in labelCounts:
        print '%s : %d %.4f' % (l, labelCounts[l], labelCounts[l]/numFiles)