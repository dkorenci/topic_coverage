from agenda.time_analysis.proportion_calc import ReadmeProportionCalc
from agenda.mapping.labeled_files import devel_all_file, test_file
from experiments.labeling.tools import *
from experiments.labeling.labelsets import *
from agenda.mapping.mapper_factory import bestMultilabelMapper
from agenda.time_analysis.analysis import corpus_time_labels_props
from agenda.mapping import labeled_files
from experiments.classification.mappers import *

import os

def getReadmePropCalc():
    trainFile = devel_all_file
    textsTrain = getLabeledTextObjectsFromParse(parseLabeledTexts(labeling_folder+trainFile))
    return ReadmeProportionCalc(textsTrain)

def readmeTimeGraph():
    propCalc = getReadmePropCalc()
    corpus_time_labels_props(propCalc, label='police brutality', saveTexts=False, saveLabels=True)

def readmeProportionsAllLabels():
    # init proportions calculator
    trainFile = devel_all_file
    textsTrain = getLabeledTextObjectsFromParse(parseLabeledTexts(labeling_folder+trainFile))
    propCalc = ReadmeProportionCalc(textsTrain)
    # run on labels
    labels = labels_rights_final()
    lab2prop = {}
    testFile = test_file
    for label in labels:
        os.chdir('/data/code/pymedialab')
        textsTest = getLabeledTextObjectsFromParse(parseLabeledTexts(labeling_folder+testFile))
        prop = propCalc.proportion(textsTest, label)
        print 'label: %s , readme proportion: %.5f' % (label, prop)
        lab2prop[label] = prop

    print lab2prop

def mapperProportionAllLabels(labeledFile=None, mapper=None):
    labels = labels_rights_final()
    mapperLabel = 'baseline_svc_devel_all_refactored'
    mapper =  getSupervisedMapper(id=mapperLabel, labels=labels, trainTexts=labeled_files.devel_all_file)
    #mapper = bestMultilabelMapper()

    labeledFile = labeled_files.test_file
    labTexts = getLabeledTextObjectsFromParse(parseLabeledTexts(labeling_folder+labeledFile))
    texts = [ txto for txto, _ in labTexts]
    N = float(len(texts))
    labelCounts = None
    for txto in texts:
        labels = mapper.map(txto)
        if labelCounts is None: labelCounts = { l:0 for l in labels }
        else:
            for l in labels: labelCounts[l] += labels[l]
    for l in labelCounts:
        print 'label: %s , mapper proportion: %.5f' % (l, labelCounts[l]/N)

propmapTest = {
    'civ.rights mov.' : (0.00505, 0.0111 , 0.0152 , 0.01875),
    'lgbt rights' : (0.05859, 0.0606 , 0.0566 , 0.05919),
    'pol. brutality' : (0.01010, 0.0152 , 0.0182 , 0.04375),
    'chapel hill' : (0.00101, 0.0010 , 0.0010 , 0.01271, 0.01),
    'reprod. rights' : (0.0, 0.0071 , 0.0121 , 0.00668),
    'viol.ag. women' : (0.00202, 0.0081 , 0.0091 , 0.02039),
    'death penalty' : (0.00202, 0.0051 , 0.0051 , 0.01570),
    'surveillance' : (0.00202, 0.0040 , 0.0061 , 0.01881),
    'gun rights' : (0.00202, 0.0071 , 0.0091 , 0.00879),
    'net neutrality' : (0.00404, 0.0061 , 0.0091 , 0.01377),
    'marijuana' : (0.01111, 0.0121 , 0.0131 , 0.02093),
    'vaccination' : (0.01515, 0.0152 , 0.0182 , 0.01799)
}

from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
def drawProportionsGraph():
    lista = ['civ.rights mov.',
        'lgbt rights' ,
        'pol. brutality',
        'chapel hill' ,
        'reprod. rights',
        'viol.ag. women',
        'death penalty',
        'surveillance',
        'gun rights',
        'net neutrality',
        'marijuana',
        'vaccination']

    x = np.arange(len(propmapTest))
    data = [[], [], [], []]
    labels = []
    for l in lista:
        for i in range(4): data[i].append(propmapTest[l][i])
        labels.append(l)

    fig, ax = plt.subplots()
    b1 = ax.bar(x + 0, data[0], color = 'darkblue', width = 0.2, hatch='')
    b2 = ax.bar(x + 0.2, data[1], color = 'darkred', width = 0.2, hatch='')
    b3 = ax.bar(x + 0.4, data[3], color = 'darkgreen', width = 0.2, hatch='')
    b4 = ax.bar(x + 0.6, data[2], color = 'wheat', width = 0.2, hatch='')
    ax.legend( (b1[0], b2[0],b3[0],b4[0]), ('SVM tagger', 'Multi Opt tagger','Readme','True proportion') )
    ax.set_xticks(x+0.4)
    ax.set_xticklabels(labels, rotation = 90)
    #ax.xaxis.set_tick_params(rotation='vertical')
    #ax.xaxis.xticks(labels, )
    plt.tick_params(axis='x', which='both', bottom='off', top='off')
    plt.tick_params(axis='y', which='both', left='off', right='off')
    plt.tight_layout(pad=0)
    #fig.savefig('/home/damir/Dropbox/science/publikacije/papers/agenda_CIKM2015/figures/props.eps', bbox_inches='tight')
    plt.show()

from math import sqrt
def calculateMAPE(prop = propmapTest, calc = [0,1,3], true = 2):
    for c in calc:
        mape = 0; rmse = 0
        for l in prop:
            pr = prop[l]
            mape += abs((pr[true]-pr[c])/pr[true])
            rmse += ((pr[true]-pr[c])*(pr[true]-pr[c]))
        mape /= len(prop)
        rmse /= len(prop); rmse = sqrt(rmse)
        print c, '%.5f' % rmse, '%.2f' % mape

if __name__ == '__main__':
    drawProportionsGraph()