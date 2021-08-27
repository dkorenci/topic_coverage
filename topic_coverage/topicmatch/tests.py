from topic_coverage.resources import pytopia_context
from topic_coverage.modelbuild.modelbuild_iter1 import loadModels, \
        uspolModelFolders, phenoModelFolders, allModelFolders, addModelsToGlobalContext
from topic_coverage.topicmatch.topicplots import valueDist
from topic_coverage.topicmatch.distance_sampling import *
from pytopia.measure.topic_distance import cosine
from topic_coverage.settings import resource_folder
from file_utils.location import FolderLocation as loc

import numpy as np, os

def testLabelParsing(tmpdir):
    opts = [
        {'pairs': testPairs, 'ints': intervals(0,1,10), 'ssize':10, 'fsize': -1 },
        {'pairs': testPairs, 'ints': intervals(0, 1, 10), 'ssize': 10, 'fsize': 1},
        {'pairs': testPairs, 'ints': intervals(0,1,10), 'ssize': 100, 'fsize': -1},
        {'pairs': testPairs, 'ints': intervals(0, 1, 10), 'ssize': 100, 'fsize': 10},
        {'pairs': testPairs, 'ints': intervals(0, 1, 10), 'ssize': 100, 'fsize': 300},
        {'pairs': testPairs, 'ints': intervals(0, 1, 10), 'ssize': 100, 'fsize': 999},
        {'pairs': testPairs, 'ints': intervals(0, 1, 10), 'ssize': 1000, 'fsize': 100},
    ]
    tmpdir = str(tmpdir)
    print 'temp.folder', tmpdir
    print 'adding models to context ...'
    addModelsToGlobalContext()
    print 'done.'
    for i, o in enumerate(opts):
        o['tmpdir'] = tmpdir
        o['seed'] = i
        labelParsing(**o)
        for f in loc(tmpdir).files():
            os.remove(f)
            print f

def labelParsing(tmpdir, pairs, ints, ssize, fsize, seed=119902):
    from topic_coverage.topicmatch.pair_labeling import \
            topicLabelText, createLabelingFiles, parseLabelingFolder, parseLabelingFile
    sample = createLabelingFiles(tmpdir, 'testSample', pairs, ints, ssize, seed, fsize)
    parsample = parseLabelingFolder(tmpdir)
    def corepair(pair): # extract core data of topic pair
        pid, t1, t2, _ = pair
        return (pid, t1, t2)
    def corelist(plist): return [ corepair(p) for p in plist ]
    sample = corelist(sample); parsample = corelist(parsample)
    assert set(sample) == set(parsample)

if __name__ == '__main__':
    pass