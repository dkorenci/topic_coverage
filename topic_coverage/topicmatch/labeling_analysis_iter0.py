from topic_coverage.resources import pytopia_context
from topic_coverage.modelbuild.modelbuild_iter1 import loadModels, \
        uspolModelFolders, phenoModelFolders, allModelFolders, addModelsToGlobalContext
from topic_coverage.topicmatch.topicplots import valueDist
from topic_coverage.topicmatch.distance_sampling import *
from pytopia.measure.topic_distance import cosine
from topic_coverage.settings import resource_folder
from file_utils.location import FolderLocation as loc

import numpy as np

def plotDist():
    valueDist([cosine], topicSetUsPol(), 5000)

def createDist():
    #createDistances(topicSetUsPol(), cosine, 356, 'test', verbose=True, sampleSize=100000)
    createDistances(topicSetUsPol(), cosine, 6356, 'uspolTopicsIter0', verbose=True)

def topicSetUsPol():
    return [t for m in loadModels(uspolModelFolders) for t in m]

def testSampling():
    pairs = loadDistances(testPairs)
    s = samplePairsByDistInterval(pairs, intervals(0, 1, 10), 20, verbose=True)
    for ival in sorted(s.keys()):
        print ival
        print len(s[ival])
        for tpd in s[ival]:
            t1, t2, d = tpd
            print '     %.4f' % d, t1, t2

def sampleUsPolIter0():
    pairs = loadDistances(iter0UspolPairs)
    s = samplePairsByDistInterval(pairs, intervals(0, 1, 10), 100, verbose=True)
    for ival in sorted(s.keys()):
        print ival
        print len(s[ival])
        for tpd in s[ival][:10]:
            _, t1, t2, d = tpd
            print '     %.4f' % d, t1, t2

def testLabeling():
    from topic_coverage.topicmatch.pair_labeling import \
            topicLabelText, createLabelingFiles, parseLabelingFolder, parseLabelingFile
    print 'adding models to context ...'
    addModelsToGlobalContext()
    print 'done.'
    lfolder = loc(resource_folder)('topicmatch', 'testlabel')
    pairs = loadDistances(testPairs)
    #print pairs[20][0]
    createLabelingFiles(lfolder, 'testSample', testPairs, intervals(0, 1, 10), 10, docs=True)
    #parseLabelingFolder(lfolder)
    #print parseLabelingFile('/datafast/topic_coverage/topicmatch/testlabel/topicPairs[0-30].txt')

if __name__ == '__main__':
    #plotDist()
    #createDist()
    #testSampling()
    #sampleUsPolIter0()
    testLabeling()