import pytopia.testing.setup
from pytopia.testing.corpora import *
from pytopia.testing.utils import createSaveLoadCompare
from pytopia.tools.parameters import flattenParams as fp, joinParams as jp

from pytopia.topic_model.ArtifTopicModel import ArtifTopicModelBuilder

import tempfile, shutil
import numpy as np

def runArtifModelSaveLoadCompare(params):
    saveDir = tempfile.mkdtemp()
    createSaveLoadCompare(ArtifTopicModelBuilder, params, saveDir)
    shutil.rmtree(saveDir, ignore_errors=True)

def randomNpmatrix(rows, cols, rseed=7834):
    np.random.seed(rseed)
    return np.random.rand(rows, cols)

def testArtifModel():
    baseparams = [{'corpus':'corpus', 'dictionary':'dict', 'text2tokens':'txt2tok'}]
    T = [50, 500]; W = [1000, 20000]; D = [None, 1000, 20000]
    matrixParams = [{ 'topicMatrix': randomNpmatrix(t, w, t*w),
                 'docTopicMatrix': (randomNpmatrix(d, t, d * t) if d else None) }
                for t in T for w in W for d in D ]
    params = jp(baseparams, matrixParams)
    for p in params:
        print 'testing', p['topicMatrix'].shape, \
              p['docTopicMatrix'].shape if p['docTopicMatrix'] is not None else None
        runArtifModelSaveLoadCompare(p)

if __name__ == '__main__' :
    testArtifModel()
