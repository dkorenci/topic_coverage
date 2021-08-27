'''
Functionality for converting labeled documents to input for ReadMe R package.
'''

import codecs
import os
import shutil

from rpy2.robjects import r as R

from pymedialab_settings.settings import object_store
from experiments.labeling.tools import *
from utils.utils import normalize_path


def createReadmeInputFromFiles(trainFile, testFile, label, outputFolder):
    if not os.path.exists(outputFolder): os.makedirs(outputFolder)
    textsTrain = getLabeledTextObjectsFromParse(parseLabeledTexts(labeling_folder+trainFile))
    textsTest = getLabeledTextObjectsFromParse(parseLabeledTexts(labeling_folder+testFile))
    createReadmeInput(textsTrain, textsTest, label, outputFolder)

def createReadmeInput(train, test, label, outFolder):
    '''
    :param train: list of (txtobject, labels)
    :param test: list of (txtobject, labels)
    '''
    outFolder = normalize_path(outFolder)
    txtList = [ (txto, lab, 1) for txto, lab in train ]
    txtList.extend([ (txto, lab, 0) for txto, lab in test ] )
    controlFile = codecs.open(outFolder+'control.txt', 'w', 'utf-8')
    sep = ' '
    controlFile.write('ROWID%sTRUTH%sTRAININGSET\n' % (sep,sep))
    for txto, lab, isTrain in txtList:
        fname = str(txto.id)+'.txt'
        f = codecs.open(outFolder+fname, 'w', 'utf-8')
        f.write(txto.text); f.close()
        if lab is None: classLabel = 0
        else:
            if label not in lab: raise Exception('labeling does not contain target label')
            classLabel = 0 if lab[label] == 0 else 1
        controlFile.write('%s%s%d%s%d\n' % (fname, sep, classLabel, sep, isTrain))

def getReadmeInputFolder():
    'create or clear and return folder to put Readme input files'
    folder = object_store+'readme_input/'
    if not os.path.exists(folder) : os.mkdir(folder)
    else:
        shutil.rmtree(folder)
        os.mkdir(folder)
    return folder

def callReadme(train, test, label, inFolder = None, rSeed=5678, numRuns=5):
    '''
    run readme to asses proportion of texts labeled with label in test,
    using train to learn estimation.
    create readme input files, call R funcionality, return label, not-label proprotions
    :param train: list of labeled Texts - (txto, labeling) pairs
    :param test: list of Texts - labeled or unlabeled
    :param label:
    :param inFolder: folder in which to put data in Readme format
    :param rSeed: random seed for R's set.seed() , must be an integer
    :param numRuns: number of Readme runs over which to average to get the final result
    :return: proportion of texts labeled with label
    '''
    # prepare input data
    x = test[0]
    if not isinstance(x, tuple): test = [(txto, None) for txto in test]
    if inFolder is None : inFolder = getReadmeInputFolder()
    createReadmeInput(train, test, label, inFolder)
    # call readme
    R.source('/data/code/pymedialab/experiments/readme/run_readme.R')
    runReadmeAvg = R['runReadmeAvg']
    #inf = R['inputFolder']
    setSeed = R['set.seed']
    setSeed(seed=rSeed)
    sol = runReadmeAvg(inputFolder=inFolder, rndseed=rSeed, runs=numRuns)
    return sol[0]
